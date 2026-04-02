import fitz

doc = fitz.open("temp_files/TIME MANAGEMENT.pdf")
for i, page in enumerate(doc):
    images = page.get_images(full=True)
    blocks = page.get_text("dict")["blocks"]
    image_blocks = [b for b in blocks if b["type"] == 1]
    print(f"Page {i+1}: get_images={len(images)}, image_blocks={len(image_blocks)}")
doc.close()