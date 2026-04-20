#!/usr/bin/env python3
"""
从 PyTorch 文档页面提取所有的 layers 和 Activations
URL: https://docs.pytorch.org/docs/stable/nn.html

依赖:
    pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict


def fetch_page(url: str) -> BeautifulSoup:
    """获取并解析网页内容"""
    print(f"正在获取页面: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')


def extract_section_items(soup: BeautifulSoup, section_title: str) -> List[Dict]:
    """提取特定章节的所有项目"""
    items = []
    
    # 查找包含章节标题的元素
    section_header = None
    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        text = tag.get_text().strip()
        if section_title.lower() in text.lower():
            section_header = tag
            break
    
    if not section_header:
        return items
    
    # 找到下一个同级或更高级的标题作为边界
    next_header = None
    for tag in section_header.find_all_next(['h2', 'h3', 'h4', 'h5']):
        if tag != section_header:
            # 检查是否是同级或更高级的标题
            tag_level = int(tag.name[1])
            header_level = int(section_header.name[1])
            if tag_level <= header_level:
                next_header = tag
                break
    
    # 在 section_header 和 next_header 之间查找所有链接
    current = section_header.next_sibling
    while current:
        if next_header and current == next_header:
            break
        
        if hasattr(current, 'find_all'):
            # 查找表格
            for table in current.find_all('table', recursive=False):
                rows = table.find_all('tr')
                for row in rows[1:]:  # 跳过表头
                    cols = row.find_all('td')
                    if len(cols) >= 1:
                        link = cols[0].find('a')
                        if link:
                            name = link.get_text().strip()
                            href = link.get('href', '')
                            description = cols[1].get_text().strip() if len(cols) > 1 else ""
                            
                            if name and 'nn.' in name or 'torch.nn' in href:
                                items.append({
                                    'name': name,
                                    'description': description,
                                    'link': href,
                                    'section': section_title
                                })
            
            # 查找 dl 标签
            for dl in current.find_all('dl', recursive=False):
                dt = dl.find('dt')
                dd = dl.find('dd')
                
                if dt:
                    link = dt.find('a')
                    if link:
                        name = link.get_text().strip()
                        href = link.get('href', '')
                        description = dd.get_text().strip() if dd else ""
                        
                        if name and ('nn.' in name or 'torch.nn' in href):
                            items.append({
                                'name': name,
                                'description': description,
                                'link': href,
                                'section': section_title
                            })
        
        current = current.next_sibling
    
    return items


def extract_from_dl(soup: BeautifulSoup, section_title: str) -> List[Dict]:
    """从 dl (description list) 标签中提取项目"""
    items = []
    
    # 查找包含章节标题的元素
    section_header = None
    for tag in soup.find_all(['h2', 'h3', 'h4']):
        if section_title.lower() in tag.get_text().lower():
            section_header = tag
            break
    
    if not section_header:
        return items
    
    # 查找该章节下的所有 dl 标签
    current = section_header
    while current:
        next_section = current.find_next_sibling(['h2', 'h3', 'h4'])
        
        for dl in current.find_all_next('dl', limit=20):
            if next_section and dl.find_previous(['h2', 'h3', 'h4']) == next_section:
                break
            
            dt = dl.find('dt')
            dd = dl.find('dd')
            
            if dt:
                link = dt.find('a')
                if link:
                    name = link.get_text().strip()
                    href = link.get('href', '')
                    description = dd.get_text().strip() if dd else ""
                    
                    items.append({
                        'name': name,
                        'description': description,
                        'link': href,
                        'section': section_title
                    })
        
        if next_section:
            break
        current = current.find_next_sibling()
    
    return items


def extract_all_layers_and_activations(soup: BeautifulSoup) -> Dict[str, List[Dict]]:
    """提取所有的 layers 和 activations，根据章节名自动分类"""
    results = defaultdict(list)
    
    # 查找所有章节标题
    all_sections = []
    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        section_title = tag.get_text().strip()
        # 只处理包含 "layer" 或 "activation" 的章节
        if 'layer' in section_title.lower() or 'activation' in section_title.lower():
            all_sections.append(section_title)
    
    print("\n=== 从章节标题提取 ===")
    for section in all_sections:
        print(f"\n处理章节: {section}")
        items = extract_section_items(soup, section)
        if not items:
            items = extract_from_dl(soup, section)
        
        if items:
            # 根据章节名分类
            section_lower = section.lower()
            if 'activation' in section_lower:
                results['activations'].extend(items)
                print(f"  找到 {len(items)} 个 activations")
            elif 'layer' in section_lower:
                results['layers'].extend(items)
                print(f"  找到 {len(items)} 个 layers")
            else:
                # 默认归为 layers
                results['layers'].extend(items)
                print(f"  找到 {len(items)} 个项目（默认归为 layers）")
        else:
            print(f"  未找到项目")
    
    return dict(results)


def extract_from_api_reference(soup: BeautifulSoup) -> Dict[str, List[Dict]]:
    """从 API 参考部分提取信息（备用方法）"""
    results = defaultdict(list)
    
    # 查找所有包含 'nn.' 或 'torch.nn' 的链接
    all_links = soup.find_all('a', href=True)
    
    seen = set()
    
    for link in all_links:
        href = link.get('href', '')
        text = link.get_text().strip()
        
        # 只处理包含 nn. 的链接
        if 'nn.' not in text and 'torch.nn' not in href:
            continue
        
        # 跳过已处理的
        if text in seen:
            continue
        
        # 获取描述
        description = ""
        parent = link.find_parent(['td', 'dt', 'li', 'p'])
        if parent:
            # 尝试从同一行的下一个单元格获取描述
            if parent.name == 'td':
                next_td = parent.find_next_sibling('td')
                if next_td:
                    description = next_td.get_text().strip()
            # 尝试从 dd 标签获取描述
            elif parent.name == 'dt':
                dd = parent.find_next_sibling('dd')
                if dd:
                    description = dd.get_text().strip()
            # 尝试从父元素的文本获取
            if not description:
                parent_text = parent.get_text().strip()
                if len(parent_text) > len(text):
                    description = parent_text.replace(text, '').strip()
        
        # 确定所属章节
        section = "Unknown"
        header = link.find_previous(['h2', 'h3', 'h4', 'h5'])
        if header:
            section = header.get_text().strip()
        
        item = {
            'name': text,
            'description': description,
            'link': href,
            'section': section
        }
        
        # 根据章节名分类：章节名包含 "layer" 的是 layers，包含 "activation" 的是 activations
        section_lower = section.lower()
        if 'activation' in section_lower:
            results['activations'].append(item)
            seen.add(text)
        elif 'layer' in section_lower:
            results['layers'].append(item)
            seen.add(text)
        elif 'nn.' in text or 'torch.nn' in href:
            # 如果章节名无法判断，默认归为 layers
            results['layers'].append(item)
            seen.add(text)
    
    return dict(results)


def main():
    url = "https://docs.pytorch.org/docs/stable/nn.html"
    
    try:
        # 获取页面
        soup = fetch_page(url)
        
        # 从章节提取
        print("\n" + "="*80)
        print("从章节标题提取")
        print("="*80)
        results = extract_all_layers_and_activations(soup)
        
        # 输出结果
        print("\n" + "="*80)
        print("提取结果汇总")
        print("="*80)
        print(f"\n总共找到 {len(results.get('layers', []))} 个 Layers")
        print(f"总共找到 {len(results.get('activations', []))} 个 Activations")
        
        # 保存为 JSON
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "pytorch_layers_activations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 保存为文本格式（便于阅读）
        output_txt = os.path.join(script_dir, "pytorch_layers_activations.txt")
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PyTorch Layers and Activations\n")
            f.write("="*80 + "\n\n")
            
            f.write("LAYERS:\n")
            f.write("-"*80 + "\n")
            for item in results.get('layers', []):
                f.write(f"\n{item['name']}\n")
                f.write(f"  Section: {item['section']}\n")
                f.write(f"  Link: {item['link']}\n")
                if item['description']:
                    f.write(f"  Description: {item['description'][:100]}...\n")
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("ACTIVATIONS:\n")
            f.write("-"*80 + "\n")
            for item in results.get('activations', []):
                f.write(f"\n{item['name']}\n")
                f.write(f"  Section: {item['section']}\n")
                f.write(f"  Link: {item['link']}\n")
                if item['description']:
                    f.write(f"  Description: {item['description'][:100]}...\n")
        
        print(f"文本格式已保存到: {output_txt}")
        
        # 打印前几个示例
        print("\n" + "="*80)
        print("示例 Layers (前5个):")
        print("="*80)
        for item in results.get('layers', [])[:5]:
            print(f"  - {item['name']} ({item['section']})")
        
        print("\n" + "="*80)
        print("示例 Activations (前5个):")
        print("="*80)
        for item in results.get('activations', [])[:5]:
            print(f"  - {item['name']} ({item['section']})")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

