import copy
import json
import os
from typing import List
import tiktoken
import srt

import datetime

from loguru import logger
import argparse
# 参数解析
parser = argparse.ArgumentParser()
# 最大的 token 输入
parser.add_argument("--max_token_input", default=350, type=int)
# 预测的 token 数量
parser.add_argument("--num_predict", default=2048, type=int)
# 温度
parser.add_argument("--temperature", default=0.1, type=float)
# 随机种子
parser.add_argument("--seed", default=42, type=int)
# ollama 翻译的自定义模型
parser.add_argument("--model", default="translate_srt:latest", type=str)
# ollama server http api url
parser.add_argument("--ollama_url", default="http://localhost:11434/api/chat", type=str)
# srt 文件传入的绝对路径
parser.add_argument("--srt_file_path", default="", type=str)
# 翻译后的 srt 文件输出的目录
parser.add_argument("--output_dir", default=".", type=str)
# 解析参数
args = parser.parse_args()


# ass 字幕内容的头
ass_00 = r"""
[Script Info]
ScaledBorderAndShadow: no
ScriptType: v4.00+
Title: Make by Github.com VideoTranslationTools
WrapStyle: 0
YCbCr Matrix: TV.709

[Custom Info]
OriginalTextStyle: 原文
TranslationTextStyle: 译文

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: NT,微软雅黑,20,&H00DEDEDE,&HF0000000,&H00111211,&H00000000,0,0,0,0,100,100,0,0,1,2,0.3,2,1,1,1,1
Style: 译文,微软雅黑,20,&H00DEDEDE,&HF0000000,&H00111211,&H00000000,0,0,0,0,100,100,0,0,1,2,0.3,2,1,1,1,1
Style: 原文,Arial Black,14,&H0062A8EB,&HF0000000,&H00111211,&H00000000,0,0,0,0,100,100,0,0,1,2,0.3,2,1,1,1,1
Style: 注释,方正艺黑_GBK,16,&H00DEDEDE,&H000000FF,&H00996150,&H00000000,0,0,0,0,100,100,0,0,1,2,0.5,8,3,3,1,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
# 一个对白的拼接部分
ass_Dialogue_0 = r"Dialogue 0,"
ass_Dialogue_1 = r",译文,,0,0,0,,"
ass_Dialogue_2 = r"\N{\r原文}"


def translate(content: str):
    import requests
    # 要发送的 JSON 数据
    data = {
        "model": args.model,
        "stream": False,
        # "prompt": content,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "options": {
            "seed": args.seed,
            # "num_predict": num_predict,
            "temperature": args.temperature
        }
    }

    # 统计耗时
    import time
    start_time = time.time()
    # 发送 POST 请求
    response = requests.post(args.ollama_url, json=data)
    end_time = time.time()
    logger.info("translate duration: {cost_time}s", cost_time=end_time - start_time)

    # 检查响应
    if response.status_code == 200:
        # 返回响应 JSON 数据
        jj = response.json()
        return 0, jj["message"]["content"]
    else:
        logger.error("发送数据失败，状态码：{status_code}", status_code=response.status_code)
        return -1, ""


# 转换到 00:15:23,222
def time_delta2str(delta):
    # 获取总秒数
    total_seconds = delta.total_seconds()
    # 将总秒数转换为时、分、秒和毫秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(delta.microseconds / 1000)
    # 格式化输出
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time


# 将 delta 时间，转换到 ass 字幕的时间格式:0:00:00.20
def time_delta2AssTimeStr(delta):
    # 获取总秒数
    total_seconds = delta.total_seconds()
    # 将总秒数转换为时、分、秒和毫秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(delta.microseconds / 100)
    # 格式化输出
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:02d}"
    return formatted_time


# 转换到 timedelta 对象
def str2time_delta(time_str):
    # 将时间字符串转换为时间间隔
    # 解析时间字符串
    parsed_time = datetime.datetime.strptime(time_str, "%H:%M:%S,%f")
    # 构造 timedelta 对象
    delta = datetime.timedelta(
        hours=parsed_time.hour,
        minutes=parsed_time.minute,
        seconds=parsed_time.second,
        milliseconds=parsed_time.microsecond // 1000,
    )
    return delta


class srt_line:
    def __init__(self, index, content, start, end):
        self.index = index
        self.content = content
        if isinstance(start, str):
            self.start = start
        else:
            self.start = time_delta2str(start)
        if isinstance(end, str):
            self.end = end
        else:
            self.end = time_delta2str(end)
        self.matched = False  # 是否匹配上原文，有可能有多句话被合并翻译后，那么有一些原文对比将省略掉了

    def json(self):
        return {
            "index": self.index,
            "content": self.content,
            "start": self.start,
            "end": self.end
        }


# 合并多个json为一个json
def merge_line2json(srt_line_jsons: []):
    out_jsons = {"lines": [], "count": len(srt_line_jsons)}
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
        start = one_line.start
        end = one_line.end
        one_line_json = srt_line(index, content, start, end)
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
        start = json_obj["start"]
        end = json_obj["end"]
        one_line_json = srt_line(index, content, start, end)
        out_lines.append(one_line_json)
    return out_lines


# 制作 ass 一句对话
def make_ass_one_line(start_delta, end_delta, zh_str, org_str):
    # ass 一句对白的开始，结束时间格式示例 0:00:00.20,0:00:02.66
    out_line = (ass_Dialogue_0 + time_delta2AssTimeStr(start_delta) + "," + time_delta2AssTimeStr(end_delta) +
                ass_Dialogue_1 + zh_str + ass_Dialogue_2 + org_str)
    return out_line + "\n"


# 制作 ass 字幕
def make_ass_file(merge_zh: List[srt.Subtitle], merge_org: List[srt.Subtitle]):

    full_ass_content = ass_00
    for index in range(len(merge_zh)):
        one_line = make_ass_one_line(merge_zh[index].start, merge_zh[index].end,
                                     merge_zh[index].content,
                                     merge_org[index].content)
        full_ass_content += one_line
    return full_ass_content


if __name__ == '__main__':

    encoding = tiktoken.get_encoding("cl100k_base")
    # srt 文件是否存在
    srt_file_path = args.srt_file_path
    if not srt_file_path:
        logger.error("srt file path is empty")
        exit(1)
    if not os.path.exists(srt_file_path):
        logger.error("srt file not exist")
        exit(1)
    srt_file = open(srt_file_path, encoding='UTF-8')
    subs = list(srt.parse(srt_file.read()))

    logger.info("subtitle lines: {lines}", lines=str(len(subs)))

    tmp_one_block = []
    wait_for_translate = []
    for i in range(len(subs)):
        tmp_one_block.append(subs[i])
        one_block_text = srt.compose(tmp_one_block)
        tmp_one_block_len = len(encoding.encode(one_block_text))
        if tmp_one_block_len > args.max_token_input:
            wait_for_translate.append(tmp_one_block)
            tmp_one_block = []
    if len(tmp_one_block) > 0:
        wait_for_translate.append(tmp_one_block)

    logger.info("will translate blocks: {blocks}", blocks=str(len(wait_for_translate)))
    # 深拷贝一份待翻译的数组
    wait_for_translate_wait_merge_zh_org = copy.deepcopy(wait_for_translate)
    # 翻译后的译文数组（这里深拷贝就是为了赋值占位）
    wait_for_translate_wait_merge_zh = copy.deepcopy(wait_for_translate)
    # 翻译前的原文数组（这里深拷贝就是为了赋值占位）
    wait_for_translate_wait_merge_org = copy.deepcopy(wait_for_translate)
    # 最后的合并结果
    merge_translated_srt = []
    merge_translated_srt_zh = []
    merge_translated_srt_org = []
    cache_index = 0
    while cache_index < len(wait_for_translate):

        logger.info("translate block: {now_index} / {all_index}",
                    now_index=str(cache_index + 1), all_index=str(len(wait_for_translate)))

        # 从 srt 数据结构构建出 srt_line 对象
        lines = convert_srt_lines_2_lines(wait_for_translate[cache_index])
        # 将 srt_line 对象转换为 json 字符串
        merged_json = merge_line2json(lines)
        # 翻译
        status, translated = translate(merged_json)

        if status == 0:
            # 提取翻译后的信息
            try:
                out_lines = parse_srt_line_jsons(translated)
            except:
                # 提取翻译后的信息失败，重新翻译
                logger.error("translate failed, parse json error: " + str(cache_index) + ", will retry")
                continue

            # 比较原始字幕和翻译后的字幕行数是否一致
            if len(wait_for_translate[cache_index]) != len(out_lines):
                # 长度不一致，重新翻译
                logger.error("translate failed, length not equal: " + str(cache_index) + ", will retry")
                continue
            else:
                # 将翻译后的字幕内容合并原始字幕，输出双语字幕
                # 当前一段翻译后的字幕，需要合并到 wait_for_translate_wait_merge 中
                for j in range(len(out_lines)):
                    # 合并字幕内容
                    merged_content = (out_lines[j].content + "\n"
                                      + wait_for_translate_wait_merge_zh_org[cache_index][j].content)
                    # 替换原始字幕内容
                    wait_for_translate_wait_merge_zh_org[cache_index][j].content = merged_content
                    # 替换原始字幕内容
                    wait_for_translate_wait_merge_zh[cache_index][j].content = out_lines[j].content

            logger.info("------------")
        else:
            logger.error("translate failed, post url error: " + str(cache_index))
            exit(1)
        # 递增索引
        cache_index += 1

    # 将 wait_for_translate_wait_merge_zh_org 加入到 merge_translated
    for i in range(len(wait_for_translate_wait_merge_zh_org)):
        merge_translated_srt.extend(wait_for_translate_wait_merge_zh_org[i])
    for i in range(len(wait_for_translate_wait_merge_zh)):
        merge_translated_srt_zh.extend(wait_for_translate_wait_merge_zh[i])
    for i in range(len(wait_for_translate_wait_merge_org)):
        merge_translated_srt_org.extend(wait_for_translate_wait_merge_org[i])

    # 如果输出目录不存在，创建目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # 保存翻译后的 srt 字幕
    with open(os.path.join(args.output_dir, "translated.srt"), "w", encoding='UTF-8') as f:
        f.write(srt.compose(merge_translated_srt))
    logger.info("translated srt saved to: {path}", path=os.path.join(args.output_dir, "translated.srt"))

    # 保存翻译后的 ass 字幕
    ass_full_content = make_ass_file(merge_translated_srt_zh, merge_translated_srt_org)
    with open(os.path.join(args.output_dir, "translated.ass"), "w", encoding='UTF-8') as f:
        f.write(ass_full_content)
    logger.info("translated srt saved to: {path}", path=os.path.join(args.output_dir, "translated.ass"))

    logger.info("done")
