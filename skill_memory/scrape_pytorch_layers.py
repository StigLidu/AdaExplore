#!/usr/bin/env python3
"""
Extract all Layers and Activations from the PyTorch documentation page.
URL: https://docs.pytorch.org/docs/stable/nn.html

Dependencies:
    pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch and parse a web page."""
    print(f"Fetching page: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')


def extract_section_items(soup: BeautifulSoup, section_title: str) -> List[Dict]:
    """Extract all items under a specific section."""
    items = []

    # Locate the element containing the section title.
    section_header = None
    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        text = tag.get_text().strip()
        if section_title.lower() in text.lower():
            section_header = tag
            break

    if not section_header:
        return items

    # Find the next sibling header at the same or higher level as the boundary.
    next_header = None
    for tag in section_header.find_all_next(['h2', 'h3', 'h4', 'h5']):
        if tag != section_header:
            # Check whether the tag is at the same or a higher level.
            tag_level = int(tag.name[1])
            header_level = int(section_header.name[1])
            if tag_level <= header_level:
                next_header = tag
                break

    # Find all links between section_header and next_header.
    current = section_header.next_sibling
    while current:
        if next_header and current == next_header:
            break

        if hasattr(current, 'find_all'):
            # Look for tables.
            for table in current.find_all('table', recursive=False):
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip the header row.
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

            # Look for <dl> tags.
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
    """Extract items from <dl> (description list) tags."""
    items = []

    # Locate the element containing the section title.
    section_header = None
    for tag in soup.find_all(['h2', 'h3', 'h4']):
        if section_title.lower() in tag.get_text().lower():
            section_header = tag
            break

    if not section_header:
        return items

    # Find all <dl> tags under that section.
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
    """Extract all layers and activations, grouped by section name."""
    results = defaultdict(list)

    # Collect all section headers.
    all_sections = []
    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        section_title = tag.get_text().strip()
        # Only keep sections whose title mentions "layer" or "activation".
        if 'layer' in section_title.lower() or 'activation' in section_title.lower():
            all_sections.append(section_title)

    print("\n=== Extracting from section headers ===")
    for section in all_sections:
        print(f"\nProcessing section: {section}")
        items = extract_section_items(soup, section)
        if not items:
            items = extract_from_dl(soup, section)

        if items:
            # Categorize based on the section title.
            section_lower = section.lower()
            if 'activation' in section_lower:
                results['activations'].extend(items)
                print(f"  Found {len(items)} activations")
            elif 'layer' in section_lower:
                results['layers'].extend(items)
                print(f"  Found {len(items)} layers")
            else:
                # Default bucket: layers.
                results['layers'].extend(items)
                print(f"  Found {len(items)} items (defaulting to layers)")
        else:
            print(f"  No items found")

    return dict(results)


def extract_from_api_reference(soup: BeautifulSoup) -> Dict[str, List[Dict]]:
    """Fallback extractor that walks the API reference section."""
    results = defaultdict(list)

    # Collect all links containing 'nn.' or 'torch.nn'.
    all_links = soup.find_all('a', href=True)

    seen = set()

    for link in all_links:
        href = link.get('href', '')
        text = link.get_text().strip()

        # Only process links that look like nn.* references.
        if 'nn.' not in text and 'torch.nn' not in href:
            continue

        # Skip duplicates.
        if text in seen:
            continue

        # Try to recover a description from nearby DOM nodes.
        description = ""
        parent = link.find_parent(['td', 'dt', 'li', 'p'])
        if parent:
            # Try to read the description from the next <td> in the same row.
            if parent.name == 'td':
                next_td = parent.find_next_sibling('td')
                if next_td:
                    description = next_td.get_text().strip()
            # Try to read the description from a sibling <dd>.
            elif parent.name == 'dt':
                dd = parent.find_next_sibling('dd')
                if dd:
                    description = dd.get_text().strip()
            # Fall back to the parent's text content.
            if not description:
                parent_text = parent.get_text().strip()
                if len(parent_text) > len(text):
                    description = parent_text.replace(text, '').strip()

        # Determine the surrounding section.
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

        # Categorize by section name: titles containing "layer" → layers, "activation" → activations.
        section_lower = section.lower()
        if 'activation' in section_lower:
            results['activations'].append(item)
            seen.add(text)
        elif 'layer' in section_lower:
            results['layers'].append(item)
            seen.add(text)
        elif 'nn.' in text or 'torch.nn' in href:
            # If the section title is unclear, default to layers.
            results['layers'].append(item)
            seen.add(text)

    return dict(results)


def main():
    url = "https://docs.pytorch.org/docs/stable/nn.html"

    try:
        # Fetch the page.
        soup = fetch_page(url)

        # Extract from section headers.
        print("\n" + "="*80)
        print("Extracting from section headers")
        print("="*80)
        results = extract_all_layers_and_activations(soup)

        # Print a summary.
        print("\n" + "="*80)
        print("Extraction summary")
        print("="*80)
        print(f"\nTotal layers found: {len(results.get('layers', []))}")
        print(f"Total activations found: {len(results.get('activations', []))}")

        # Write JSON output.
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "pytorch_layers_activations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

        # Write a human-readable text dump.
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

        print(f"Text format saved to: {output_txt}")

        # Print a few sample entries.
        print("\n" + "="*80)
        print("Sample layers (first 5):")
        print("="*80)
        for item in results.get('layers', [])[:5]:
            print(f"  - {item['name']} ({item['section']})")

        print("\n" + "="*80)
        print("Sample activations (first 5):")
        print("="*80)
        for item in results.get('activations', [])[:5]:
            print(f"  - {item['name']} ({item['section']})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
