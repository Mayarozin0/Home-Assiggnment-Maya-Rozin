import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from phase1.ocr_extractor import process_form


def main():
    st.title("National Insurance Institute Form Extractor")
    st.write("Upload a form to extract information")

    uploaded_file = st.file_uploader("Upload a form (PDF/JPG)", type=["pdf", "jpg", "jpeg"])

    if uploaded_file:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Process the file and extract information
        with st.spinner("Processing form..."):
            try:
                result = process_form(temp_file_path)

                # Display the extracted data
                st.success("Form processed successfully!")

                # Show validation results
                if result["validation_results"]["missing_required_fields"] or result["validation_results"][
                    "format_issues"]:
                    st.warning("Validation issues detected:")
                    if result["validation_results"]["missing_required_fields"]:
                        st.write("Missing required fields:",
                                 ", ".join(result["validation_results"]["missing_required_fields"]))
                    if result["validation_results"]["format_issues"]:
                        st.write("Format issues:", ", ".join(result["validation_results"]["format_issues"]))
                else:
                    st.success("All data validated successfully!")

                # Display the structured data
                st.json(result["form_data"])

                # Option to download as JSON
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(result["form_data"], indent=2, ensure_ascii=False),
                    file_name="extracted_form_data.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error processing form: {str(e)}")

            # Clean up the temporary file
            os.unlink(temp_file_path)


if __name__ == "__main__":
    main()