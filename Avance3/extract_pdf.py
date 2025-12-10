import pypdf

pdf_path = "Enunciado proyecto Semestral IA.pdf"
output_path = "enunciado.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for page in reader.pages:
            f.write(page.extract_text() + "\n\n")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
