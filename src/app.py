import os
import base64
import json
import pandas as pd
import gradio as gr
from openai import OpenAI
from PIL import Image
import io
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Optional
import time
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English model
print('OCR loaded')
# 初始化 OpenAI 客户端（阿里云百炼）
client = OpenAI(
    api_key='sk-5ed0a549b0274a1d8506a04e61c4d5e3',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def keep_rois_and_white_background(
    image_np: np.ndarray,
    rec_boxes: List[List[int]],
    output_path: Optional[str] = None,
    save_to_local: bool = False
) -> np.ndarray:
    """
    保留指定 ROI 区域，其余部分设为白色背景。
    
    Args:
        image_np: 输入图像 (H, W, 3)，RGB，np.uint8
        rec_boxes: 文字区域的边界框列表，每个 box 格式为 [x_min, y_min, x_max, y_max]
        output_path: 保存路径（如果 save_to_local=True）
        save_to_local: 是否保存到本地
    
    Returns:
        处理后的图像 (H, W, 3)，np.ndarray，RGB
    """
    h, w, c = image_np.shape
    # 创建全白色背景图像
    result_img = np.full((h, w, c), 255, dtype=np.uint8)

    for box in rec_boxes:
        x_min, y_min, x_max, y_max = box

        # 边界检查
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_min >= x_max or y_min >= y_max:
            continue

        # 复制 ROI 区域
        result_img[y_min:y_max, x_min:x_max] = image_np[y_min:y_max, x_min:x_max]

    # 保存到本地（使用 PIL）
    if save_to_local and output_path:
        # 转为 PIL Image 并保存
        pil_image = Image.fromarray(result_img)  # 默认是 RGB
        pil_image.save(output_path, "JPEG", quality=95, optimize=True)

    return result_img

# 图像转 Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

from PIL import Image
import io


def compress_image(
    img: Image.Image,
    max_size_kb: int = 1024,
    max_dim: int = 1920,
    quality_step: int = 5,
    min_quality: int = 50,
) -> Image.Image:
    """智能压缩图像（保持画质，减小体积）"""
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg

    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    quality = 95

    while quality >= min_quality:
        output.seek(0)
        output.truncate(0)
        img.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
        size_kb = output.tell() / 1024
        if size_kb <= max_size_kb:
            break
        quality -= quality_step

    if quality < min_quality:
        output.seek(0)
        output.truncate(0)
        img.save(output, format="JPEG", quality=min_quality, optimize=True)

    output.seek(0)
    compressed_img = Image.open(output)
    return compressed_img, output.tell()  # 返回图像 + 压缩后字节大小


# 定义每种文档类型的默认提取字段模板
field_templates = {
    "Passport": [
        "full_name", "full_name_english", "passport_number", "nationality",
        "date_of_birth", "gender", "issue_date", "expiry_date","Add your own keyword..."
    ],
    "IDCard": [
        "name", "gender", "ethnicity", "date_of_birth", "address", "id_number","Add your own keyword..."
    ],
    "DriverLicense": [
        "name", "gender", "nationality", "address", "date_of_birth",
        "first_issue_date", "vehicle_classes", "validity_period","Add your own keyword..."
    ],
    "BankCard": [
        "card_number", "bank_name", "cardholder_name", "expiration_date","Add your own keyword..."
    ],
    "Receipt": [
        "merchant_name", "receipt_no", "transaction_date", "total_amount", "currency","Add your own keyword..."
    ]
}

# 根据 doc_type 和用户字段生成 Prompt
def build_prompt(extract_fields):
    if not extract_fields:
        return "请尽可能提取图像中的所有关键信息，并以JSON格式返回。"

    fields_str = ", ".join([f"`{f}`" for f in extract_fields])
    return (
        f"Please extract the following fields from the document image: {fields_str}. "
        "Return a JSON object with these exact keys. "
        "If any field is missing or unreadable, set its value to an empty string. "
        "Do not add any extra fields."
    )

# 初始化表格（editable）
def init_table(doc_type):
    fields = field_templates.get(doc_type, [])
    df = pd.DataFrame({
        "Field": fields,
        "Value": [""] * len(fields)
    })
    return df

def pil_to_base64_in_memory(image: Image.Image, format="JPEG", quality=85) -> str:
    """将 PIL 图像直接转为 base64，不保存本地文件"""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=format, quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def ndarray_to_base64(image_np: np.ndarray, quality: int = 90) -> str:
    """
    将 PaddleOCR 输出的 numpy 图像快速转为 base64 字符串（JPEG）
    假设输入一定是: RGB, uint8, HWC, 无透明通道
    """
    # 直接转 PIL
    pil_image = Image.fromarray(image_np)
    
    # 内存中编码为 JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# 提取信息函数
def extract_info(image, doc_type, user_table):
    if image is None:
        return "❌ Please upload an image", user_table

    # 获取要提取的字段（只提取 Field 列中非空的）
    try:
        extract_fields = user_table["Field"].dropna().astype(str).tolist()
        extract_fields.remove("Add your own keyword...")
        if not extract_fields:
            return "⚠️ No fields to extract (Field column is empty)", user_table
    except Exception as e:
        return f"⚠️ Invalid table format {e}", user_table

    # ✅ 1. 压缩图像（在内存中）
    img_copy = image.copy()
    temp_path_cp = "temp_image.jpg"
    img_copy.save(temp_path_cp)

    
    compressed_img, compressed_size = compress_image(image, max_size_kb=512)
    original_size = len(pil_to_base64_in_memory(image)) * 3 // 4  # Base64 编码后约增大 1/3

    temp_path_cp = "temp_image_comprsd.jpg"
    compressed_img.save(temp_path_cp)
    
    print(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes")
    print('start ocr')
    img_array = np.array(img_copy)
    compressed_img_array = np.array(compressed_img)
    result = ocr.predict(compressed_img_array)

    unwrap_img = result[0]['doc_preprocessor_res']['output_img']
    if result and result[0]:
        rec_boxes = result[0]['rec_boxes']
        compressed_img_roi = keep_rois_and_white_background(unwrap_img, rec_boxes,'rio.jpg',True)

    # ✅ 2. 直接将 PIL 图像转为 base64（不保存文件）
    base64_image = ndarray_to_base64(compressed_img_roi)
    # base64_image = pil_to_base64_in_memory(compressed_img)

    # 构造 Prompt（基于用户定义的字段）
    prompt = build_prompt(extract_fields)

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[
                {"role": "system", "content": "You are a professional document information extraction assistant. Return only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )


        # ✅ 请求完成，记录结束时间
        end_time = time.time()

        # ✅ 提取服务器时间戳
        server_timestamp = completion.created  # Unix 时间戳（秒）
        local_request_time = start_time

        # ✅ 计算各种耗时
        total_end_to_end = end_time - start_time           # 本地总耗时（发出 → 收到）
        time_to_server = server_timestamp - start_time     # 网络 + 服务器排队时间
        server_processing = total_end_to_end - time_to_server  # 估算服务器处理时间

        # ✅ 打印详细时间分析
        print("\n" + "="*50)
        print("⏱️  PERFORMANCE METRICS")
        print(f"Local Request Time : {start_time:.3f} ({time.strftime('%H:%M:%S', time.localtime(start_time))})")
        print(f"Server Created Time: {server_timestamp} ({time.strftime('%H:%M:%S', time.localtime(server_timestamp))})")
        print(f"Total End-to-End   : {total_end_to_end:.3f} seconds")
        print(f"Network + Queue    : {time_to_server:.3f} seconds")
        print(f"Server Processing  : {server_processing:.3f} seconds")
        print("="*50)

        content = completion.choices[0].message.content.strip()
        result_json = json.loads(content)

        # ✅ 新增：打印完整返回结果（用于调试）
        print("\n" + "="*50)
        print("🟢 Raw Completion Response:")
        print(completion)

        # ✅ 新增：打印 usage 信息（重点）
        print("\n📊 Token Usage:")
        print(f"   Prompt Tokens:      {completion.usage.prompt_tokens}")
        print(f"   Completion Tokens:  {completion.usage.completion_tokens}")
        print(f"   Total Tokens:       {completion.usage.total_tokens}")
        if completion.usage.prompt_tokens_details:
            print(f"   Cached Tokens:      {completion.usage.prompt_tokens_details.cached_tokens}")

        # 更新表格：只更新用户定义的字段
        updated_rows = []
        for _, row in user_table.iterrows():
            key = str(row["Field"])
            value = result_json.get(key, "") if pd.notna(row["Field"]) else ""
            updated_rows.append({"Field": key, "Value": value})

        df_result = pd.DataFrame(updated_rows)
        print('df_result', df_result)
        print('OK')
        return "✅ Extraction successful", df_result

    except Exception as e:
        error_msg = f"❌ Extraction failed: {str(e)}"
        print(error_msg)
        return error_msg, user_table

# 当文档类型变化时，重置表格
def on_doc_type_change(doc_type):
    return init_table(doc_type)

# Gradio Interface
with gr.Blocks(title="Document Information Extractor") as demo:
    gr.Markdown("# 📄 Document Information Extraction System")
    gr.Markdown("Upload an image and select document type. Edit the table to customize fields to extract.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            doc_type = gr.Dropdown(
                choices=list(field_templates.keys()),
                value="Passport",
                label="Document Type"
            )
            submit_btn = gr.Button("🔍 Extract Information", variant="primary")

        with gr.Column(scale=1):
            status = gr.Textbox(label="Status")
            # 可编辑表格：用户可修改 Field 和 Value
            dataframe_input = gr.Dataframe(
                label="Fields to Extract (Editable)",
                headers=["Field", "Value"],
                value=lambda: init_table("Passport"),  # 默认加载 Passport 字段
                interactive=True,
                wrap=True,
                type="pandas"
            )

    # 绑定事件
    doc_type.change(
        fn=on_doc_type_change,
        inputs=doc_type,
        outputs=dataframe_input
    )

    submit_btn.click(
        fn=extract_info,
        inputs=[image_input, doc_type, dataframe_input],
        outputs=[status, dataframe_input]
    )

# 启动
if __name__ == "__main__":
    demo.launch()