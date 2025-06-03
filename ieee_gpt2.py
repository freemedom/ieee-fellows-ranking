import csv, json, os, time
from pathlib import Path
# from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
from tqdm import tqdm

limit_year = 2015
infile  = Path("fellows_full.tsv")# Path("fellows.tsv")
api_key = "替换自己的key"

# load_dotenv()
# client = OpenAI()           # 会自动读取 OPENAI_API_KEY
client = OpenAI(
    api_key=api_key
)

SYSTEM = """
给定 IEEE Fellow 的姓名列表，返回这些人"国籍（或长期工作国家）"和"长期任职的院校"的列表，返回英文。\
输出必须是严格的 JSON 数组，列表的每个对象形如：
{"name":"···","country":"···","university":"···"}\
不要输出其它文字。若不确定就让字段值为空字符串。"""

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(6))
def ask_batch(batch, model="gpt-4o-mini"):
    prompt = "姓名列表：" + ", ".join(batch)
    while True:
        rsp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system", "content":SYSTEM},
                {"role":"user", "content":prompt}
            ]
        )
        data = json.loads(rsp.choices[0].message.content)
        # 如果返回的dict不是只有一个key，说明格式不对，需要重试
        if isinstance(data, dict) and len(data) != 1:
            print(f"⚠️ API返回格式不正确（key数量：{len(data)}），重试中...")
            continue
        return data

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main(model="gpt-4.1"): # 用gpt-4o-mini gpt-4.1-mini 得到的数据都不对 换成gpt-4.1对了，价格贵5倍

    outfile = Path("fellows_out.csv")

    with infile.open("r", encoding="utf-8") as fin:
        rows   = list(csv.DictReader(fin, delimiter="\t"))
        # 只保留2010年及之后的Fellow
        rows   = [r for r in rows if int(r["Year"]) >= limit_year]
        names  = [r["Fellow"] for r in rows]

    results = {}
    for batch in tqdm(list(chunks(names, 10))): # 这里将names列表每10个名字分成一组，用于批量调用GPT API。
        data = ask_batch(batch, model)   # &rarr; list[dict]
        # 获取字典中唯一的key 
        result_key = next(iter(data))
        for item in data[result_key]:
            results[item["name"]] = item

    # 写回
    fieldnames = {'Fellow', 'Year', 'Country', 'University', 'Citation'} # rows[0].keys() | {"Country","University"} # {'Country', 'Citation', 'Fellow', 'Year', 'University'}
    with outfile.open("w", newline="", encoding="utf-8") as fout:
        wr = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=",")
        wr.writeheader()
        for r in rows:
            meta = results.get(r["Fellow"], {})
            r["Country"]    = meta.get("country","")
            r["University"] = meta.get("university","")
            wr.writerow(r)

    print(f"✅ 已保存 {outfile}（共 {len(rows)} 行）")

def fill_missing(model="gpt-4.1"): # gpt-4.1-mini
    file = Path("fellows_out.csv")
    
    # 读取数据
    with file.open("r", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin, delimiter=","))
    
    # 找到Country或University为空的记录
    missing_rows = [r for r in rows if not r["Country"] or not r["University"]]
    
    if not missing_rows:
        print("✅ 没有需要补充的数据")
        return False
    
    print(f"找到 {len(missing_rows)} 条需要补充的数据")
    
    # 获取需要查询的名字
    names = [r["Fellow"] for r in missing_rows]
    
    # 批量查询
    results = {}
    for batch in tqdm(list(chunks(names, 10))):
        data = ask_batch(batch, model)
        # 获取字典中唯一的key
        result_key = next(iter(data))
        for item in data[result_key]:
            results[item["name"]] = item
    
    # 更新数据
    for row in rows:
        if not row["Country"] or not row["University"]:
            meta = results.get(row["Fellow"], {})
            if not row["Country"]:
                row["Country"] = meta.get("country", "")
            if not row["University"]:
                row["University"] = meta.get("university", "")
    
    # 写回文件
    fieldnames = ['Fellow', 'Year', 'Country', 'University', 'Citation']
    with file.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ 已更新文件 {file}")
    
    # 检查是否还有缺失
    with file.open("r", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin, delimiter=","))
    still_missing = [r for r in rows if not r["Country"] or not r["University"]]
    if still_missing:
        print(f"⚠️ 仍有 {len(still_missing)} 条数据未补充完整")
        return True
    else:
        print("✅ 所有数据已补充完整")
        return False

if __name__ == "__main__":
    main()
    print("\n开始补充缺失数据...\n")
    loop_count = 0
    while fill_missing() and loop_count < 10:
        loop_count += 1
        print(f"\n继续补充剩余数据... (第{loop_count}次循环)\n")
    
    if loop_count >= 10:
        print("\n已达到最大循环次数(10次)，停止补充数据。")
    
    
    
