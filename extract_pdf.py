import pdfplumber
import sys

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"d:\IMACS\precipitation-nowcasting\THOR-Khang-Hoai-An-Minh20250612.pdf"
output_path = r"d:\IMACS\precipitation-nowcasting\THOR_paper_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    all_text = []
    for i, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text:
            all_text.append(f"=== PAGE {i} ===\n{text}")
    
    full_text = "\n\n".join(all_text)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Extracted {len(pdf.pages)} pages to {output_path}")
    print("\n" + "="*50)
    print("CONTENT PREVIEW:")
    print("="*50 + "\n")
    print(full_text)
