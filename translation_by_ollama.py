import copy
import json

import tiktoken
import srt

max_token_input = 350
num_predict = 2048
temperature = 0.1

# model = "starling-lm:alpha"
# model = "starling-lm:latest"
# model = "gemma:7b"
# model = "llama2-chinese:13b"
# model = "llama2:13b"
# model = "qwen:7b"
# model = "translate_srt_gemma:latest"
model = "translate_srt:latest"
# model = "translate_srt_llama3:latest"
# model = "translate_srt_commandr:latest"
# model = "translate_srt_llama2-chinese"

system_content = '''你是一位精通视频字幕翻译的专业人员，请你帮我将字幕文件srt格式的文本内容段落翻译成简体中文。
你需要结合对话上下文来翻译，考虑到字幕对话中有口语化的描述，合理翻译到简体中文，保持原有格式，不要遗漏任何信息，保证每一句对话都是完整输出的。
给你多少个对话，你就要输出多少个对话，要按每个对话一个段落的方式输出翻译后的内容，不要将多个对话合并到一个段落中。如果给你的对话的语言是英文，那么这个对话的
只需要输出翻译后的内容，不要附加额外的内容。你可以在脑子里预输出翻译后的内容，然后你检查一下是否满足 srt 字幕输出的格式，不满足你就要重新整理好，满足后再输出内容'''  # 现在请按照上面的要求从第一行开始翻译以下内容为简体中文：

user_content = '''1
00:00:01,001 --> 00:00:02,001
hello world
2
00:00:03,003 --> 00:00:04,004
Sergeant James Betzo, NYPD.'''  # 这里是示例问题
assistant_content = '''1
00:00:01,001 --> 00:00:02,001
hello world
你好世界
2
00:00:03,003 --> 00:00:04,004
Sergeant James Betzo, NYPD.
纽约市警察局（NYPD）警长詹姆斯·贝特索（James Betzo）。'''  # 这里是示例答案

err_text = '''198
00:12:32,205 --> 00:12:34,946
Dude, his diet is steroids and Johnny Walker blocks.

199
00:12:34,946 --> 00:12:35,726
There you go.

200
00:12:35,726 --> 00:12:36,327
Two fingers every day.

201
00:12:36,327 --> 00:12:37,147
Two fingers.

202
00:12:37,147 --> 00:12:37,987
Please.

203
00:12:37,987 --> 00:12:38,607
You should have four.

204
00:12:38,607 --> 00:12:40,768
You earned four, for God's sake.

205
00:12:40,768 --> 00:12:43,609
I'm the luckiest.

206
00:12:43,609 --> 00:12:44,409
I say it all the time.

207
00:12:44,409 --> 00:12:45,469
I'm the luckiest guy out there.

208
00:12:45,469 --> 00:12:47,150
I really am, you know?

209
00:12:47,150 --> 00:12:48,190
This is gonna be some day, though.

210
00:12:48,190 --> 00:12:49,010
This is nice.

211
00:12:49,010 --> 00:12:50,391
Yeah.

212
00:12:50,391 --> 00:12:55,372
Sergeant James Betzo, NYPD.

213
00:12:55,372 --> 00:12:58,373
Joseph T. Callahan, FDNY.

214
00:13:03,453 --> 00:13:08,034
Keith M. Laughlin, FDNY.

215
00:13:08,034 --> 00:13:16,236
Charles S. Zou, FDNY.

216
00:13:16,236 --> 00:13:18,676
This event was over the top.

217
00:13:18,676 --> 00:13:24,337
I've been to several, but like I said, any time you get to tell a story, it's a good day.

218
00:13:24,337 --> 00:13:31,979
Anybody who wants to get a selfie or a Jon Stewart autograph, please wait until the media finishes.

219
00:13:33,925 --> 00:13:40,450
Anybody who wants to have their park or lawn designed and manicured by John Feel also come up.
'''


def translate(content: str):
    import requests
    # API 端点
    # url = "http://192.168.8.135:11434/api/generate"
    url = "http://localhost:11434/api/chat"
    # url = "http://192.168.8.135:11434/api/chat"
    # 要发送的 JSON 数据
    data = {
        "model": model,
        "stream": False,
        # "prompt": content,
        "messages": [
            # {
            #     "role": "system",
            #     "content": system_content
            # },
            # {
            #     "role": "user",
            #     "content": user_content
            # },
            # {
            #     "role": "assistant",
            #     "content": assistant_content
            # },
            {
                "role": "user",
                "content": content
            }
        ],
        "options": {
            "seed": 42,
            # "num_predict": num_predict,
            "temperature": temperature
        }
    }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 检查响应
    if response.status_code == 200:
        # 返回响应 JSON 数据
        jj = response.json()
        print("now translate total duration:", jj["total_duration"] / 1000000000, "s")
        print("now translate total tokens:", str(jj["eval_count"]))
        return 0, jj["message"]["content"]
    else:
        print("发送数据失败，状态码：", response.status_code)
        return -1, ""


class srt_line:
    def __init__(self, index, content):
        self.index = index
        self.content = content

    def json(self):
        return {
            "index": self.index,
            "content": self.content
        }


# 合并多个json为一个json
def merge_line2json(srt_line_jsons: []):
    out_jsons = {"lines": []}
    for one_line in srt_line_jsons:
        out_jsons["lines"].append(one_line.json())
    # dict to json
    json_str = json.dumps(out_jsons, indent=None, ensure_ascii=False)
    return json_str


def convert_srt_lines_2_lines(str_blocks: []):
    out_jsons = []
    for one_line in str_blocks:
        index = one_line.index
        content = one_line.content
        one_line_json = srt_line(index, content)
        out_jsons.append(one_line_json)
    return out_jsons


# 使用正则表达式提取字符串中的 json 字符串数组
def extract_json_str(text: str):
    # import re
    # pattern = re.compile(r'\{.*\}')
    # result = pattern.findall(text)
    # return result
    import re
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, text)
    json_strings = [match.strip() for match in matches]
    return json_strings


# 解析字符串获取其中的 srt_line json 对象
def parse_srt_line_jsons(json_str: str):
    json_lines = extract_json_str(json_str)
    out_lines = []
    for one_line in json_lines:
        json_obj = json.loads(one_line)
        index = json_obj["index"]
        content = json_obj["content"]
        one_line_json = srt_line(index, content)
        out_lines.append(one_line_json)
    return out_lines


if __name__ == '__main__':

    encoding = tiktoken.get_encoding("cl100k_base")

    # status, translated = translate(err_text)

    srt_file_path = r"D:\Subtitle\No Responders Left Behind (2021) WEBRip-1080p_whisperx_2.srt"
    srt_file = open(srt_file_path, encoding='UTF-8')
    subs = list(srt.parse(srt_file.read()))
    print("sub lines:" + str(len(subs)))
    tmp_one_block = []
    wait_for_translate = []
    for i in range(len(subs)):
        tmp_one_block.append(subs[i])
        one_block_text = srt.compose(tmp_one_block)
        tmp_one_block_len = len(encoding.encode(one_block_text))
        if tmp_one_block_len > max_token_input:
            wait_for_translate.append(tmp_one_block)
            tmp_one_block = []
    if len(tmp_one_block) > 0:
        wait_for_translate.append(tmp_one_block)

    print("will translate blocks:" + str(len(wait_for_translate)))
    # 深拷贝一份待翻译的数组
    wait_for_translate_wait_merge = copy.deepcopy(wait_for_translate)
    #
    merge_translated = []
    for i in range(len(wait_for_translate)):

        print("translate block:" + str(i + 1) + "/" + str(len(wait_for_translate)))
        # 从 srt 数据结构构建出 srt_line 对象
        lines = convert_srt_lines_2_lines(wait_for_translate[i])
        # 将 srt_line 对象转换为 json 字符串
        merged_json = merge_line2json(lines)
        # 翻译
        status, translated = translate(merged_json)
        if status == 0:
            # 提取翻译后的信息
            out_lines = parse_srt_line_jsons(translated)
            # 比较原始字幕和翻译后的字幕行数是否一致
            if len(wait_for_translate[i]) != len(out_lines):
                print("translate failed:" + str(i))
                exit(1)
            # 将翻译后的字幕内容合并原始字幕，输出双语字幕
            # 当前一段翻译后的字幕，需要合并到 wait_for_translate_wait_merge 中
            for j in range(len(out_lines)):
                # 合并字幕内容
                merged_content = (out_lines[j].content + "\n"
                                  + wait_for_translate_wait_merge[i][j].content)
                # 替换原始字幕内容
                wait_for_translate_wait_merge[i][j].content = merged_content
            print("------------")
        else:
            print("translate failed:" + str(i))
            exit(1)

    # 将 wait_for_translate_wait_merge 加入到 merge_translated
    for i in range(len(wait_for_translate_wait_merge)):
        merge_translated.extend(wait_for_translate_wait_merge[i])
    # 写出翻译好的字幕
    with open("translated.srt", "w", encoding='UTF-8') as f:
        f.write(srt.compose(merge_translated))
    print("done")
