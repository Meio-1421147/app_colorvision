import base64
import io
import json

from PIL import Image
from dotenv import load_dotenv  # .envファイル読み込み
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai.errors import APIError

# .env ファイルを読み込み、GEMINI_API_KEYを環境変数にセット
load_dotenv()

app = Flask(__name__)

# Gemini APIクライアントの初期化
# google-genaiライブラリは環境変数 GEMINI_API_KEY を自動的に読み込む
try:
    client = genai.Client()
except Exception as e:
    print("FATAL ERROR: Gemini Client initialization failed. Check if GEMINI_API_KEY is set correctly.")
    client = None

# モデル名
MODEL_NAME = "gemini-2.5-flash"


@app.route('/')
def index():
    """フロントエンドのHTMLをレンダリングします"""
    return render_template('index.html')


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """画像と音声/テキストコマンドを受け取り、Gemini APIで解析して結果を返すエンドポイント"""

    if client is None:
        return jsonify({"error": "Gemini API client is not initialized. Check API Key."}), 500

    # 1. データ取得
    data = request.json
    image_data_url = data.get('image')
    command = data.get('command', '').strip()

    # 🚨 データチェック
    if not image_data_url:
        return jsonify({"error": "No image data received"}), 400

    # 2. Base64デコード処理
    try:
        # data:image/jpeg;base64, のヘッダー部分を無視
        header, encoded = image_data_url.split(',', 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))
    except Exception as e:
        return jsonify({"error": f"Base64 decoding error: {str(e)}"}), 500

    # 3. PIL Imageオブジェクト作成処理 (img_pilを定義)
    try:
        # image_bytes は前の try ブロックで定義されている
        img_pil = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        # Image.open または convert でエラーが発生した場合
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500

    # 4. Gemini APIへの指示（プロンプトエンジニアリング）
    prompt_text = f"""
    あなたは画像内の物体を検出し、ユーザーの指示に基づいて結果をフィルタリングし、JSON形式で正確なバウンディングボックス座標を返すエキスパートです。

    【ユーザーの指示（最も重要な条件）】: "{command}"

    【要件】:
    1.  画像内の物体を検出し、**ユーザーの指示（色と物）に合致するもの**のみを選んでください。
    2.  各オブジェクトの**バウンディングボックス**は、画像の正規化座標（0.0から1.0）として `[x_min, y_min, x_max, y_max]` の配列形式で出力してください。
    3.  検出された各オブジェクトの**支配的な色**をHexコード（例: #FF0000）で出力してください。
    4.  出力は必ず以下のJSONスキーマに従う、単一のJSON配列としてください。他のテキストや説明は一切含めないでください。

    JSONスキーマ:
    [
      {{"box": [0.1, 0.2, 0.3, 0.4], "name": "...", "color_hex": "#RRGGBB"}},
      ...
    ]
    """

    # 5. API呼び出し
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_text, img_pil],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        json_output = json.loads(response.text)

        return jsonify({"success": True, "detections": json_output, "command": command})

    except APIError as e:
        # 対策1: デバッグのためプロンプトの一部をログに出力
        print(f"Failed prompt (API Error): {prompt_text[:200]}...")
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Gemini Raw Output: {response.text}")
        return jsonify({"error": "Gemini did not return valid JSON or format was incorrect."}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # 実行前に pip install Flask google-genai Pillow python-dotenv が必要です
    app.run(debug=True, threaded=True)