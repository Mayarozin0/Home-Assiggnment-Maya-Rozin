import os
import json
from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import re
import math

def ocr_extractor(file_path) -> dict:
    """
    Extracts text and selection marks from a form using Azure Document Intelligence
    """
    load_dotenv()

    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

    # Create the client
    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Load file
    with open(file_path, "rb") as f:
        document_bytes = f.read()
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            document_bytes,
            pages="1" # only process page 1
        )
        result = poller.result()

    # Extract text and selection marks from the result
    extracted_data = extract_text_and_marks(result)
    return extracted_data

def clean_number(text):
    """Cleans numbers from unwanted characters"""
    # Keep only digits, decimal points, and hyphens (removing spaces)
    cleaned = re.sub(r'[^\w\.\-\s]', '', text)
    # Remove spaces between digits
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned

def compute_center(points):
    """Compute the average (x, y) as the 'center' of the polygon."""
    avg_x = sum(p['x'] for p in points) / len(points)
    avg_y = sum(p['y'] for p in points) / len(points)
    return (avg_x, avg_y)


def match_key_with_checkbox(lines, marks):
    """
    lines: list of dict, each with 'content' and 'polygon'
    marks: list of dict, each with 'state' and 'position' (the polygon)
    """
    matches = []

    for line in lines:
        line_center = compute_center(line['position'])
        lx, ly = line_center

        # Filter marks that are to the right of the line center
        valid_marks = []
        for mark in marks:
            mark_center = compute_center(mark['position'])
            mx, my = mark_center
            if mx > lx:  # only consider marks to the right
                valid_marks.append((mark, mark_center))

        # Find the closest among the valid marks
        best_mark = None
        best_dist = float('inf')
        for (mark, mark_center) in valid_marks:
            dist = math.dist(line_center, mark_center)
            if dist < best_dist:
                best_dist = dist
                best_mark = mark

        matches.append({
            "line_content": line['content'],
            "matched_mark_state": best_mark['state'] if best_mark else None,
        })

    return matches


def extract_text_and_marks(result) -> dict:
    """
    Extract text and selection marks from the OCR result
    """
    extracted_text = ""
    keys_for_selection_marks = []
    patterns = [
        "נקבה", "זכר", "במפעל", "ת. דרכים בעבודה", "ת. דרכים בדרך לעבודה/מהעבודה",
        "תאונה בדרך ללא רכב", "אחר", "הנפגע חבר בקופת חולים", "כללית", "מאוחדת",
        "מכבי", "לאומית", "הנפגע אינו חבר בקופת חולים", "מהות התאונה"
    ]

    # Iterate over the OCR pages
    for page in result.pages:
        if page.page_number == 1:  # Only process page 1
            for line in page.lines:
                line_text = line.content

                # 1) Check if this line matches any of the patterns
                has_pattern = any(re.search(pattern, line_text) for pattern in patterns)
                if has_pattern:
                    # If the line is not the special "טופס זה מנוסח..." line, add it to keys_for_selection_marks
                    if line_text != "טופס זה מנוסח בלשון זכר אך פונה לנשים וגברים כאחד":
                        keys_for_selection_marks.append({
                            "content": line_text,
                            "position": [
                                {"x": p.x, "y": p.y} for p in line.polygon
                            ] if hasattr(line, 'polygon') else []
                        })
                    # Important: Skip adding it to extracted_text
                    continue

                # 2) If the line doesn't match a pattern, proceed with normal text extraction
                if re.search(r'\d', line_text):
                    # Clean content if it has digits
                    cleaned_content = clean_number(line_text)
                    extracted_text += cleaned_content + "\n"
                else:
                    # Keep it as-is
                    extracted_text += line_text + "\n"

    # Extract selection marks (checkboxes)
    selection_marks = []
    for page in result.pages:
        if page.page_number == 1:
            for mark in page.selection_marks:
                # Create a simpler representation of each selection mark
                selection_marks.append({
                    "state": mark.state,  # "selected" or "unselected"
                    "content": mark.content if hasattr(mark, 'content') else "",
                    "position": [{"x": p.x, "y": p.y} for p in mark.polygon] if hasattr(mark, 'polygon') else []
                })


    # find the matching lines and selection marks
    matches = match_key_with_checkbox(keys_for_selection_marks, selection_marks)

    # Return both text and selection marks
    return {
        "text": extracted_text,
        "selection_marks": matches
    }

def extract_form_data_with_openai(extracted_data):
    """
    Use Azure OpenAI to extract data from the form with prior knowledge about form structure
    """
    load_dotenv()

    # Get OpenAI configuration from environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = "gpt-4o"
    deployment = "gpt-4o"

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint
    )

    # Desired JSON structure for output
    desired_json_structure = """
    {
      "lastName": "",
      "firstName": "",
      "idNumber": "",
      "gender": "",
      "dateOfBirth": {
        "day": "",
        "month": "",
        "year": ""
      },
      "address": {
        "street": "",
        "houseNumber": "",
        "entrance": "",
        "apartment": "",
        "city": "",
        "postalCode": "",
        "poBox": ""
      },
      "landlinePhone": "",
      "mobilePhone": "",
      "jobType": "",
      "dateOfInjury": {
        "day": "",
        "month": "",
        "year": ""
      },
      "timeOfInjury": "",
      "accidentLocation": "",
      "accidentAddress": "",
      "accidentDescription": "",
      "injuredBodyPart": "",
      "signature": "",
      "formFillingDate": {
        "day": "",
        "month": "",
        "year": ""
      },
      "formReceiptDateAtClinic": {
        "day": "",
        "month": "",
        "year": ""
      },
      "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": ""
      }
    }
    """

    # Create prompt with prior knowledge about form structure
    prompt = f"""
    You are an expert in extracting information from National Insurance Institute forms (ביטוח לאומי) in Hebrew.

    The following text is from a form requesting medical treatment for a self-employed worker injured at work (Form BL/283).

    Form text:
    {extracted_data['text']}

    Checkbox states in the form (when state=selected, the checkbox is checked):
    {json.dumps(extracted_data['selection_marks'], ensure_ascii=False)}

    Important information about the form structure:
    1. Gender (gender): The form has two options - "זכר" (male) and "נקבה" (female). Extract the value based on which checkbox is selected.
    2. Accident Location (accidentLocation): The form has several options - "במפעל" (at workplace), "ת. דרכים בעבודה" (traffic accident at work), "ת. דרכים בדרך לעבודה/מהעבודה" (traffic accident on the way to/from work), "תאונה בדרך ללא רכב" (non-vehicle accident on the way), "אחר" (other). Extract the selected option.
    3. Health Fund (healthFundMember): The form has four options - "כללית" (Clalit), "מאוחדת" (Meuhedet), "מכבי" (Maccabi), "לאומית" (Leumit). Extract the selected option.

    Important guidance for date fields:
    The form has multiple date fields which follow a specific structure and position in the form:
    1. Form Filling Date (תאריך מילוי הטופס): Appears first in the form, near section #1.
    2. Form Receipt Date at Clinic (תאריך קבלת הטופס בקופה): Appears in first section if the form, after the Form Filling Date.
    3. Date of Birth (תאריך לידה): Appears in the personal details section, typically with the person's name and ID.
    4. Date of Injury (תאריך הפגיעה):  Appears in the bottom part of the form, typically in the section of accident details.

    Appears in the bottom part of the form, typically before section #5.

    For each date field, look for numbers in format DD/MM/YYYY or numbers separated specifically into day, month, and year fields.
    Look for dates appearing in these specific contexts within the form to correctly assign them.

    Extract all relevant information from the form and return it in the following JSON structure:
    {desired_json_structure}

    For fields that do not appear or cannot be extracted, use an empty string.
    Pay special attention to:
    1. Correctly extracting gender, accident location, and health fund according to the selected checkboxes
    2. Properly breaking down dates into day, month, and year - make sure to match each date with its correct field based on its position and context in the form
    3. Correctly extracting address details
    4. For date formats, extract numbers appearing in boxes or fields marked as day (יום), month (חודש), and year (שנה)

    Do not invent information that does not exist in the form. If an information item does not appear, use an empty string.
    """

    # Call Azure OpenAI
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system",
             "content": "You are an expert assistant in processing forms and documents in Hebrew submitted to the National Insurance Institute."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        top_p=0.4,
        response_format={"type": "json_object"}  # Ensure response is in JSON
    )

    # Extract JSON from the response
    try:
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        return result_json
    except json.JSONDecodeError:
        # If parsing fails, return the raw text for debugging
        return {"error": "Failed to parse OpenAI response", "raw_response": response.choices[0].message.content}


def validate_extraction(data):
    """
    Validate the extracted data for completeness and correctness.
    Returns validation results and a confidence score.
    """
    validation_results = {
        "missing_required_fields": [],
        "format_issues": [],
        "consistency_issues": [],
        "field_stats": {"total": 0, "filled": 0, "empty": 0}
    }

    # Required fields check
    required_fields = ["lastName", "firstName", "idNumber", "dateOfInjury",
                       "timeOfInjury", "accidentDescription", "injuredBodyPart"]
    for field in required_fields:
        if not data.get(field):
            validation_results["missing_required_fields"].append(field)

    # Count fields for completeness statistics
    flat_fields = flatten_dict(data)
    validation_results["field_stats"]["total"] = len(flat_fields)

    for key, value in flat_fields.items():
        if value and value != "":
            validation_results["field_stats"]["filled"] += 1
        else:
            validation_results["field_stats"]["empty"] += 1

    # Format validations
    # ID number check
    if data.get("idNumber"):
        if not data["idNumber"].isdigit() or len(data["idNumber"]) != 9:
            validation_results["format_issues"].append("idNumber should be 9 digits")

    # Date format checks
    date_fields = ["dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"]
    for field in date_fields:
        if field in data and data[field]:
            if not all(data[field].get(part) for part in ["day", "month", "year"]):
                validation_results["format_issues"].append(f"{field} is incomplete")
            else:
                # Check if date parts are numeric and in valid ranges
                try:
                    day = int(data[field]["day"])
                    month = int(data[field]["month"])
                    year = int(data[field]["year"])

                    if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
                        validation_results["format_issues"].append(f"{field} has invalid date range")
                except (ValueError, TypeError):
                    validation_results["format_issues"].append(f"{field} has non-numeric components")

    # Phone number validations
    for phone_field in ["landlinePhone", "mobilePhone"]:
        if data.get(phone_field):
            digits_only = ''.join(c for c in data[phone_field] if c.isdigit())
            if len(digits_only) < 7 or len(digits_only) > 10:
                validation_results["format_issues"].append(f"{phone_field} has invalid format")

    # Consistency checks
    if data.get("dateOfInjury") and data.get("formFillingDate"):
        injury_date = get_date_value(data["dateOfInjury"])
        filling_date = get_date_value(data["formFillingDate"])

        if injury_date and filling_date and injury_date > filling_date:
            validation_results["consistency_issues"].append("Date of injury is after form filling date")

    # Check consistency between accident description and injured body part
    if data.get("accidentDescription") and data.get("injuredBodyPart"):
        if len(data["accidentDescription"]) < 10:
            validation_results["consistency_issues"].append("Accident description seems too short")

    # Check that form receipt date at clinic is after or equal to form filling date
    if data.get("formFillingDate") and data.get("formReceiptDateAtClinic"):
        filling_date = get_date_value(data["formFillingDate"])
        receipt_date = get_date_value(data["formReceiptDateAtClinic"])

        if filling_date and receipt_date and receipt_date < filling_date:
            validation_results["consistency_issues"].append("Form receipt date at clinic is before form filling date")

    # Language checks for text fields
    text_fields = ["lastName", "firstName", "accidentDescription", "accidentLocation",
                   "injuredBodyPart"]
    for field in text_fields:
        if data.get(field) and len(data[field]) > 0:
            detected_language = detect_language(data[field])
            if detected_language not in ["hebrew", "english", "mixed"]:
                validation_results["format_issues"].append(f"{field} language seems invalid")

    # Calculate completeness percentage
    completeness = 0
    if validation_results["field_stats"]["total"] > 0:
        completeness = (validation_results["field_stats"]["filled"] /
                        validation_results["field_stats"]["total"]) * 100

    # Calculate confidence score
    confidence_score = calculate_confidence_score(validation_results, completeness)

    validation_results["completeness_percentage"] = completeness
    validation_results["confidence_score"] = confidence_score

    return validation_results


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_date_value(date_dict):
    """Convert a date dict to a comparable value"""
    try:
        if all(date_dict.get(part) for part in ["day", "month", "year"]):
            # Convert all parts to integers and create a comparable value
            day = int(date_dict["day"])
            month = int(date_dict["month"])
            year = int(date_dict["year"])
            return year * 10000 + month * 100 + day
    except (ValueError, TypeError) as e:
        print(f"Error converting date: {e}")
    return None


def detect_language(text):
    """Simple language detection for Hebrew/English"""
    hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    english_chars = sum(1 for c in text if ('a' <= c.lower() <= 'z'))

    if hebrew_chars > 0 and english_chars > 0:
        return "mixed"
    elif hebrew_chars > 0:
        return "hebrew"
    elif english_chars > 0:
        return "english"
    else:
        return "unknown"


def calculate_confidence_score(validation_results, completeness):
    """Calculate an overall confidence score based on validation results"""
    score = 100  # Start at 100%

    # Deduct for missing required fields
    score -= len(validation_results["missing_required_fields"]) * 10

    # Deduct for format and consistency issues
    score -= len(validation_results["format_issues"]) * 5
    score -= len(validation_results["consistency_issues"]) * 3

    # Factor in completeness (weighted at 30% of total score)
    score = (score * 0.7) + (completeness * 0.3)

    # Cap between 0 and 100
    return max(0, min(100, score))


def process_form(file_path):
    """
    Main function to process a form: extract OCR data, extract form fields, and validate
    """
    # Extract OCR data
    extracted_data = ocr_extractor(file_path)
    print("debug:", extracted_data)
    # Extract form fields using OpenAI
    form_data = extract_form_data_with_openai(extracted_data)

    # Validate the extracted data
    validation_results = validate_extraction(form_data)

    return {
        "form_data": form_data,
        "validation_results": validation_results
    }

