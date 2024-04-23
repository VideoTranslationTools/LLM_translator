import argparse
import json
import os
import pickle
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 参数解析
parser = argparse.ArgumentParser()

proxies = {'http': 'http://127.0.0.1:10809', 'https': 'http://127.0.0.1:10809'}
max_new_tokens = 3000

split_text = "<start_of_turn>model"
replace_text = "<eos>"

# prompt_pre = "你现在是一名专业的翻译人员，我会提供一段需要翻译文本给你，你需要帮我翻译成简体中文，注意保持每句话之间的换行格式，下面是需要翻译的内容：\n"
prompt_pre = "你现在是一名专业的翻译人员，我会提供一段需要翻译文本给你，你需要帮我翻译成简体中文，注意保持每句话之间的换行格式，比如 1#: 2#: 这样的前缀。下面是需要翻译的内容：\n"


def encode_index(content: str):
    """""
    content 根据换行符进行分割:
    I mean, this is sci-fi shit. This is CIA.\n
    This is Area 51! - Rudy, focus. What do you mean?\n
    转换为：
    1#：I mean, this is sci-fi shit. This is CIA.\n
    2#：This is Area 51! - Rudy, focus. What do you mean?\n
    """""
    str_list = content.split("\n")
    i = 1
    out_content = ""
    for str_line in str_list:
        if str_line == "":
            continue
        out_content += str(i) + "#:" + str_line + "\n"
        i += 1

    return out_content


def decode_index(content: str):
    """""
    content 根据换行符进行分割:
    1#：I mean, this is sci-fi shit. This is CIA.\n
    2#：This is Area 51! - Rudy, focus. What do you mean?\n
    转换为：
    I mean, this is sci-fi shit. This is CIA.\n
    This is Area 51! - Rudy, focus. What do you mean?\n
    """""
    str_list = content.split("\n")
    out_content = ""
    for str_line in str_list:
        if str_line == "":
            continue
        one_line_list = str_line.split("#:")
        out_content += one_line_list[1].strip().replace(replace_text, "") + "\n"

    return out_content


def decode_generate_result(content: str):
    temp_content = content.split(split_text)
    return temp_content[1]


# 生成对话模板
def generate_chat_template(content: str):
    tmp = prompt_pre + content
    chat = [
        {"role": "user", "content": tmp},
    ]
    return chat


def generate_chat_template_2(content: str):
    tmp = prompt_pre + content
    return tmp


# 检查文件的处理状态
def is_file_processed(save_dir_path: str, file_name: str) -> bool:
    """
    Check whether the given file has been processed based on the file_status.json file.

    :param save_dir_path: The directory path where the JSON file is located.
    :param file_name: The name of the binary file to check the status.
    :return: True if the file is processed, False if not processed or not found.
    """

    # Load the file status from the JSON file
    with open(os.path.join(save_dir_path, 'file_status.json'), 'r', encoding='utf-8') as f:
        file_status = json.load(f)

    # Return the file processing status if the file exists in the file status dictionary
    return file_status.get(file_name, False)


# 更新文件的处理状态
def update_file_status(save_dir_path: str, file_name: str, status: bool):
    """
    Update the processing status of a file in the file_status.json file.

    :param save_dir_path: The directory path where the JSON file is located.
    :param file_name: The name of the binary file to update the status.
    :param status: The new processing status (True: processed, False: not processed).
    """

    # Load the file status from the JSON file
    with open(os.path.join(save_dir_path, 'file_status.json'), 'r', encoding='utf-8') as f:
        file_status = json.load(f)

    # Update the file status
    file_status[file_name] = status

    # Save the updated file status back to the JSON file
    with open(os.path.join(save_dir_path, 'file_status.json'), 'w', encoding='utf-8') as f:
        json.dump(file_status, f, ensure_ascii=False, indent=4)


def chat_processor(model_name: str, t, m, temp_content: str):

    if model_name == "MiniCPM":
        responds, history = m.chat(tokenizer, generate_chat_template_2(temp_content), temperature=0.8, top_p=0.8)
        decode_text = responds
    elif model_name == "Gemma":
        prompt = t.apply_chat_template(generate_chat_template(temp_content),
                                       tokenize=False,
                                       add_generation_prompt=True)

        inputs = t.encode(prompt, add_special_tokens=True, return_tensors="pt")
        outputs = m.generate(input_ids=inputs.to("cuda"), temperature=0.8, top_p=0.8, max_new_tokens=max_new_tokens)
        out_text = tokenizer.decode(outputs[0])
        decode_text = decode_generate_result(out_text)
    else:
        raise Exception("model_name 参数错误")

    return decode_text


def translator(model_name: str, t, m, need_translate_content_list: list):
    print("-----------------------------")
    # 统计推理耗时
    start = time.time()
    need_translate_content = ""
    for tt in need_translate_content_list:
        need_translate_content += tt + "\n"

    temp_content = encode_index(need_translate_content)

    decode_text = chat_processor(model_name, t, m, temp_content)

    end = time.time()
    print("generate 耗时：" + str(end - start) + "秒")

    return decode_index(decode_text)


def translator_srt_parts_dir(model_name: str, t, m, save_subtitle_parts_dir: str):
    start_time = time.time()
    # 枚举出 save_subtitle_parts_dir 这个文件夹下面，所有 need_translate_parts pkl 后缀名的文件
    need_translate_parts = []
    count = 0
    for root, dirs, files in os.walk(save_subtitle_parts_dir):
        for file in files:
            if file.endswith(".pkl") and file.startswith("need_translate_parts"):
                pass
            else:
                continue
            print("now file name: " + file)
            count += 1
            print("count: " + str(count))
            file_f_path = os.path.join(root, file)
            need_translate_parts.append(file_f_path)
            if is_file_processed(save_subtitle_parts_dir, file):
                continue
            with open(file_f_path, 'rb') as f:
                sen_list = pickle.load(f)
            # 统计执行的时间
            print('------------------------------------')
            print('Translating: ', file)

            # 进行翻译
            translated_content = translator(model_name, t, m, sen_list)
            # 剔除 translated_content 中的空白行
            translated_content = "\n".join([line for line in translated_content.split("\n") if line.strip() != ""])
            # 如果翻译成功了，那么就标记这个文件已经完成处理
            if translated_content is not None:
                # 将翻译的结果保存下来
                translated_result_file = file.replace("need_translate_parts", "translated_result_parts")
                translated_result_file = os.path.join(save_subtitle_parts_dir, translated_result_file)
                with open(translated_result_file, 'wb') as f:
                    pickle.dump(translated_content, f)

                update_file_status(save_subtitle_parts_dir, file, True)

                print('Need Translated: ', file)
                print('Translated Result: ', translated_result_file)
            else:
                print('Failed to translate: ', file)

    print('------------------------------------')
    end_time = time.time()
    print('translator_srt_parts_dir Time cost: ', end_time - start_time)


def init_model(model_name: str):
    # 耗时统计，加载模型，统计到毫秒
    start = time.time()

    if model_name == "MiniCPM":
        torch.manual_seed(0)
        path = 'openbmb/MiniCPM-2B-sft-bf16'
        in_tokenizer = AutoTokenizer.from_pretrained(path)
        in_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda',
                                                        trust_remote_code=True)
    elif model_name == "Gemma":
        # model_id = "google/gemma-2b-it"
        # tokenizer = AutoTokenizer.from_pretrained(model_id)  # , proxies=proxies
        # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")  # , proxies=proxies

        # 使用 int8 load_in_8bit load_in_4bit
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 下面的代码是示例如何从本地缓存钟加载模型
        # 目标是找到 config.json 这个文件
        cache_dir = (r"C:\Users\allan716\.cache\huggingface\hub\models--google--gemma-2b-it\snapshots"
                     r"\9642e777f24fde593d204a9b2471dce33334e64a")
        in_tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True)  # , proxies=proxies
        in_model = AutoModelForCausalLM.from_pretrained(cache_dir, trust_remote_code=True,
                                                        # quantization_config=quantization_config,  # 使用 int8
                                                        # torch_dtype=torch.float16,  # 使用 float16
                                                        device_map="auto")  # , proxies=proxies
    else:
        raise Exception("model_name 参数错误")

    end = time.time()
    print("加载模型耗时：" + str(end - start) + "秒")
    return in_tokenizer, in_model


if __name__ == '__main__':
    # 下面的代码是从 huggingface 下载模型，可能需要使用登录，access_token 登录
    # from huggingface_hub import login
    # 登录以便能够下载模型
    # login()
    use_model_name = "MiniCPM"

    tokenizer, model = init_model(use_model_name)

    translator_srt_parts_dir(use_model_name, tokenizer, model,
                             r"C:\WorkSpace\PythonThings\deepl_translate_subtitles\test_datas\No Responders Left "
                             r"Behind (2021) WEBRip-1080p_whisperx_2")
    # 解析输出 outputs，只需要 model 回复的内容
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("response: " + response)
    print("All Done.")
