from dotenv import load_dotenv
from app_py import markdown_extractor, describe_page_images, build_full_context
load_dotenv()

pdf_path = "temp_files/GNS 103 SUMMARY.pdf"

pages = markdown_extractor(pdf_path)
descriptions = describe_page_images(pdf_path)
context = build_full_context(pages, descriptions)