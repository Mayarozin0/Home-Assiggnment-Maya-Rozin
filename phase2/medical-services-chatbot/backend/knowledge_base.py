import os
import json
from bs4 import BeautifulSoup
import re
from pathlib import Path


def extract_html_content(html_file):
    """Extract structured content from HTML file."""
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')

        # Extract title (main category)
        category = soup.find('h2').text.strip()

        # Extract description (all paragraphs before the table)
        description_elements = []
        for elem in soup.find('h2').next_siblings:
            if elem.name == 'table':
                break
            if elem.name == 'p':
                description_elements.append(elem.text.strip())

        description = ' '.join(description_elements)

        # Extract services from table
        services = []
        table = soup.find('table')
        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            cells = row.find_all('td')
            service_name = cells[0].text.strip()

            # Extract data for each HMO and tier
            hmo_data = {
                'maccabi': parse_hmo_cell(cells[1]),
                'meuhedet': parse_hmo_cell(cells[2]),
                'clalit': parse_hmo_cell(cells[3])
            }

            services.append({
                'name': service_name,
                'hmo_data': hmo_data
            })

        # Extract contact information based on section headings
        contact_info = {}

        # Find all h3 headings that might contain contact information
        contact_headings = soup.find_all('h3')
        for heading in contact_headings:
            heading_text = heading.text.strip()
            contact_list = heading.find_next('ul')

            if contact_list:
                # Create a dictionary for this specific contact section
                section_contacts = {}
                for item in contact_list.find_all('li'):
                    text = item.text.strip()
                    if 'מכבי' in text:
                        section_contacts['maccabi'] = text.replace('מכבי:', '').strip()
                    elif 'מאוחדת' in text:
                        section_contacts['meuhedet'] = text.replace('מאוחדת:', '').strip()
                    elif 'כללית' in text:
                        section_contacts['clalit'] = text.replace('כללית:', '').strip()

                # Add this section to the contact_info with the heading as the key
                if section_contacts:
                    contact_info[heading_text] = section_contacts

        # Extract additional info
        additional_info = {}
        additional_section = soup.find('h3', string=re.compile(r'לפרטים נוספים'))
        if additional_section:
            info_list = additional_section.find_next('ul')
            if info_list:
                for item in info_list.find_all('li'):
                    if 'מכבי' in item.text:
                        additional_info['maccabi'] = extract_additional_info(item)
                    elif 'מאוחדת' in item.text:
                        additional_info['meuhedet'] = extract_additional_info(item)
                    elif 'כללית' in item.text:
                        additional_info['clalit'] = extract_additional_info(item)

        return {
            'category': category,
            'description': description,
            'services': services,
            'contact_info': contact_info,
            'additional_info': additional_info
        }


def parse_hmo_cell(cell):
    """Parse HMO cell to extract tier information."""
    text = cell.text.strip()
    tiers = {
        'gold': {'hebrew': 'זהב', 'data': None},
        'silver': {'hebrew': 'כסף', 'data': None},
        'bronze': {'hebrew': 'ארד', 'data': None}
    }

    # Extract information for each tier
    for tier_en, tier_data in tiers.items():
        tier_heb = tier_data['hebrew']
        pattern = re.compile(
            f"{tier_heb}:(.*?)(?:{list(tiers.values())[0]['hebrew']}:|{list(tiers.values())[1]['hebrew']}:|{list(tiers.values())[2]['hebrew']}:|$)",
            re.DOTALL)
        match = pattern.search(text)
        if match:
            tiers[tier_en]['data'] = match.group(1).strip()

    return {
        'gold': tiers['gold']['data'],
        'silver': tiers['silver']['data'],
        'bronze': tiers['bronze']['data']
    }


def extract_additional_info(item):
    """Extract additional information like phone and website."""
    item_text = item.text.strip()
    info = {}

    # Extract phone
    phone_match = re.search(r'טלפון:(.*?)(?:\n|$)', item_text)
    if phone_match:
        info['phone'] = phone_match.group(1).strip()

    # Extract website
    link = item.find('a')
    if link:
        info['website'] = link.get('href')
        info['website_text'] = link.text.strip()

    return info


def create_json_files(data, output_dir, filename_base):
    """Create separate JSON files for each HMO and tier."""
    hmos = {
        'maccabi': 'מכבי',
        'meuhedet': 'מאוחדת',
        'clalit': 'כללית'
    }

    tiers = {
        'gold': 'זהב',
        'silver': 'כסף',
        'bronze': 'ארד'
    }

    for hmo_en, hmo_heb in hmos.items():
        hmo_dir = os.path.join(output_dir, filename_base, hmo_en)
        os.makedirs(hmo_dir, exist_ok=True)

        for tier_en, tier_heb in tiers.items():
            # Create a JSON object specific to this HMO and tier
            hmo_tier_data = {
                'category': data['category'],
                'description': data['description'],
                'hmo': hmo_heb,
                'tier': tier_heb,
                'services': []
            }

            # Add services specific to this HMO and tier
            for service in data['services']:
                service_data = service['hmo_data'][hmo_en]
                if service_data and service_data[tier_en]:
                    hmo_tier_data['services'].append({
                        'name': service['name'],
                        'benefits': service_data[tier_en]
                    })

            # Add contact information from the appointment section with the original heading
            for section_name, section_data in data['contact_info'].items():
                if "מספרי טלפון" in section_name and hmo_en in section_data:
                    if 'contact' not in hmo_tier_data:
                        hmo_tier_data['contact'] = {}

                    # Use the original section name as the field name
                    field_name = section_name.replace(' ', '_')
                    hmo_tier_data['contact'][field_name] = section_data[hmo_en]


            # Add additional information with separate phone and website fields
            if hmo_en in data['additional_info']:
                if 'contact' not in hmo_tier_data:
                    hmo_tier_data['contact'] = {}

                additional = data['additional_info'][hmo_en]
                if 'phone' in additional:
                    hmo_tier_data['contact']['phone'] = additional['phone']
                if 'website' in additional:
                    hmo_tier_data['contact']['website'] = additional['website']

            # Write to JSON file
            json_file = os.path.join(hmo_dir, f"{tier_en}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(hmo_tier_data, f, ensure_ascii=False, indent=4)

            print(f"Created {json_file}")


def process_html_files(html_dir, output_dir):
    """Process all HTML files in the directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for html_file in os.listdir(html_dir):
        if html_file.endswith('.html'):
            file_path = os.path.join(html_dir, html_file)
            print(f"Processing {file_path}...")

            # Extract base filename without extension
            filename_base = os.path.splitext(html_file)[0]

            # Parse HTML
            data = extract_html_content(file_path)

            # Create JSON files
            create_json_files(data, output_dir, filename_base)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    html_dir = BASE_DIR.parent / 'data/phase2_data'
    output_dir =  BASE_DIR.parent / 'data/processed_data'
    process_html_files(html_dir, output_dir)

