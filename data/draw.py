from pdf2image import convert_from_path
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def pdf_to_image(pdf_path, page_num, output_path, scale=0.5):
    """将 PDF 的指定页面转换为图像，并缩放"""
    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
    image = images[0]
    image = image.resize((int(image.width * scale), int(image.height * scale)))  # 缩放图像
    image.save(output_path)


def combine_pdfs_to_single_page(input_pdfs, output_pdf, layout='3x2', scale=0.5):
    """
    将多个 PDF 文件的第一页合并到一个 PDF 的单页中，并进行缩放。

    :param input_pdfs: 输入 PDF 文件路径列表
    :param output_pdf: 输出 PDF 文件路径
    :param layout: 布局模式 '3x2' 或 '2x3'
    :param scale: 缩放因子（0.5表示将图像尺寸缩小到原始的一半）
    """
    # 确定布局
    rows, cols = (3, 2) if layout == '3x2' else (2, 3)
    page_width, page_height = letter
    margin = 0.5 * inch
    cell_width = (page_width - 2 * margin) / cols
    cell_height = (page_height - 2 * margin) / rows

    # 创建 PDF 文件
    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))

    # 处理每个 PDF
    for i, pdf_path in enumerate(input_pdfs):
        if i >= rows * cols:
            break  # 超出布局限制
        row, col = divmod(i, cols)

        # 将 PDF 转为图像，并进行缩放
        image_path = f"temp_image_{i + 1}.png"
        pdf_to_image(pdf_path, 0, image_path, scale=scale)

        # 打开图像并获取尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size

        # 计算位置和缩放
        cell_width = 285
        cell_height = 160
        x = margin + col * cell_width
        y = page_height - margin - (row + 1) * cell_height
        c.drawImage(image_path, x, y, width=cell_width, height=cell_height)

    # 保存 PDF 文件
    c.save()

# 示例用法
input_pdfs = ['disease_cure.pdf', 'disease_feedback.pdf', 'symptom.pdf', 'patient_cure_num.pdf', 'patient_feedback_num.pdf', 'symptom_num.pdf']
input_pdfs = ['../figs/' + elem for elem in input_pdfs]
output_pdf = 'combined_charts_single_page.pdf'
combine_pdfs_to_single_page(input_pdfs, output_pdf, layout='3x2', scale=0.5)