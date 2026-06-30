#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
国家统计局新版 API 批量抓取农业类指标数据脚本
- 合并保存为 CSV（utf-8-sig）
- 使用 requests.Session + 重试策略
- 支持 tqdm（若未安装则降级为普通打印）
- 请求间隔：1-2 秒随机延时
- 配置集中在顶部

保存文件：fetch_stats_agri.py
运行：python fetch_stats_agri.py
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------------------- 顶部配置（方便修改） --------------------
BASE_URL = "https://data.stats.gov.cn/dg/website/publicrelease/web/external"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests",
    "Content-Type": "application/json;charset=UTF-8",
}
DEFAULT_YEAR_RANGE = "2005YY-2024YY"
ROOT_ID = "c4d82af16c3d4f0cb4f09d4af7d5888e"
SHOW_TYPE = "1"
DEFAULT_AREA = "000000000000"  # 全国 12 位代码
# 省份代码与名称映射（6 位省份代码 -> 名称）
PROVINCE_MAP = {
    "110000": "北京", "120000": "天津", "130000": "河北", "140000": "山西",
    "150000": "内蒙古", "210000": "辽宁", "220000": "吉林", "230000": "黑龙江",
    "310000": "上海", "320000": "江苏", "330000": "浙江", "340000": "安徽",
    "350000": "福建", "360000": "江西", "370000": "山东", "410000": "河南",
    "420000": "湖北", "430000": "湖南", "440000": "广东", "450000": "广西",
    "460000": "海南", "500000": "重庆", "510000": "四川", "520000": "贵州",
    "530000": "云南", "540000": "西藏", "610000": "陕西", "620000": "甘肃",
    "630000": "青海", "640000": "宁夏", "650000": "新疆"
}

# 是否每个数据集单独保存 CSV（默认 False，生成一个合并文件）
SAVE_PER_DATASET = False

# 指标批量请求的分块大小（避免 payload 过大）
INDICATOR_BATCH_SIZE = 300

# 请求超时（秒）
REQUEST_TIMEOUT = 60

# 输出目录
OUTPUT_DIR = Path("output_stats_agri")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集内置（全部为分省数据，按用户要求替换）
DATASETS = [
    # ==================== 价格指数 ====================
    {"name": "农村居民消费价格分类指数", "cid": "87049adcbe26478c828ddf6a9d5d306c", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村商品零售价格指数", "cid": "d9ae0312467c425cb49e5bbdbf37d042", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农业生产资料价格分类指数", "cid": "a65038653f0a4558addb8c8d074e34a8", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农产品生产价格指数", "cid": "02e95baab5e2448e80cc3b8cbf5cf2cf", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    # ==================== 资源与环境 ====================
    {"name": "森林资源", "cid": "1f8851c9c718459d8040e7cae3b70f14", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "造林面积", "cid": "ede2e88747254872b1388b940436b51b", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "林业重点工程造林面积", "cid": "9aeeaeeea4734e57802cc40265dafd84", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "草原建设利用", "cid": "edc68a2ff88743b8aa45e1a8349020ac", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "自然灾害", "cid": "1cb6524fd270419ca681d72a3f815f39", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "森林火灾", "cid": "a514a0f407364388a51368cc8fbd40d6", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "森林病害虫害防治", "cid": "f07565aedd8d4935be47aae7ec3a4b42", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "林业投资资金来源", "cid": "e8620b421afc43c897489471ea5c1e6f", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "林业投资", "cid": "ce415b8620714d49baf3beb071c50a24", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    # ==================== 农业（农村基本情况与农业生产） ====================
    {"name": "农村基层组织情况", "cid": "308da8e9cccf4e85bb81ad87e25ecbcf", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "乡村户数和乡村人口", "cid": "28604900dfff48698f8109e08ae1ef97", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "乡村从业人员", "cid": "9d23a6bbf77b4a6f8d46c2c5a44c45a8", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农林牧渔业总产值及指数", "cid": "61daa3937e1546bf911c3a9dff70cb3f", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "主要农业机械年末拥有量", "cid": "95979d6ab4944c8c978c0a7f6f9671c5", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "有效灌溉面积农用化肥施用量农村水电站及用电量", "cid": "ac37a55cdaad4d0ba433e5a0ababc620", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村水电建设和发电量", "cid": "b7ba28a680ed4bab96fa865d55b20b13", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "水利设施和除涝面积", "cid": "a59c4ea36a7c44248ef563f438faf886", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农用柴油和农药使用量", "cid": "b4ac2a8421ef4f8f88617776e2ed15b8", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农用塑料薄膜使用量", "cid": "a0b866e19b6345b2b50d8a7cb79a2e81", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村居民家庭生产性固定资产原值", "cid": "a07176824e474044af9faead8e1891dc", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村居民家庭平均每百户拥有主要生产性固定资产数量", "cid": "f06f58508dcd47cd8cf77a0a2c617a21", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村居民家庭土地经营情况", "cid": "d89747a0b82c47d294f7957047243eb2", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "主要农作物播种面积", "cid": "7b69fe6179314779bd45def65ab9e1d9", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "主要农作物单位面积产量", "cid": "a151036e4904405b8d8199b37f21d450", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "果园面积", "cid": "8592dfade68b41ed8f88d32ca46bb003", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "瓜果类面积", "cid": "10bafda7c59c48f5a0336e6f9e05d1df", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "茶园面积", "cid": "80520bbae4f64b09a4cb9986b6158d14", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "茶叶水果产量", "cid": "78cc8d301460404fa7969dc0728c71d1", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "瓜果类单位面积产量", "cid": "3b80741828654889aa3c18d47f5453e9", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "主要林产品产量", "cid": "35ba239db914508a408ae138f5da348", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "牲畜饲养", "cid": "3c993de07d1949c8bcd7e62aa2836fd8", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "牲畜出栏量", "cid": "f6c93e8ddfe54f6db554d6f5dd02fb86", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "畜产品产量", "cid": "170452ef17404055ab198aec4d1327ae", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "水产品产量", "cid": "6f7377a8bbdb47ccb9de7067be0f73d3", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村居民家庭平均每人出售主要农产品", "cid": "b6ae2afcf62b449b94bee6c2c0fd35f1", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "农村居民家庭平均每人出售主要畜产品及水产品", "cid": "14d5bbc9424e4757aa9dfa8ef7469e29", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "按人口平均的主要农产品产量", "cid": "2fdc92cf06c64dff90b38a687098d8b4", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "平均每一农业劳动力生产的主要农产品", "cid": "fe84754fcb9b4bd885c20d114e768784", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "水产养殖面积", "cid": "4f396a24b5564f2e8436179790cec552", "year_range": "2005YY-2024YY", "region_type": "provincial"},
    {"name": "受灾和成灾面积", "cid": "7b8b80ac9a94088b3550ddbc3e0b33d", "year_range": "2005YY-2024YY", "region_type": "provincial"},
]

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -------------------- 工具函数 --------------------
def create_session_with_retries(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """
    创建带重试策略的 requests.Session
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_kwargs = dict(total=total_retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504])
    try:
        retry = Retry(**retry_kwargs, allowed_methods=frozenset(["GET", "POST"]))
    except TypeError:
        retry = Retry(**retry_kwargs, method_whitelist=frozenset(["GET", "POST"]))

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sanitize_filename(s: str) -> str:
    """
    将字符串转换为安全文件名
    """
    invalid = '<>:\\"/\\|?*\n\r\t'
    res = "".join(c for c in s if c not in invalid)
    return res.strip().replace(" ", "_")


def split_batches(lst: List[Any], n: int):
    """
    将列表切分为大小为 n 的批次
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def try_sleep(min_s: float = 1.0, max_s: float = 2.0):
    """
    随机延时，避免压垮服务器
    """
    t = random.uniform(min_s, max_s)
    time.sleep(t)


def normalize_area_code(code: str) -> str:
    """
    将 6 位省份代码或 12 位代码规范成 12 位地区代码
    例如 '110000' -> '110000000000'
    """
    s = str(code)
    if len(s) == 12:
        return s
    if len(s) == 6:
        return s + "000000"
    # 兜底：填充到 12 位
    return s.ljust(12, "0")


# -------------------- API 交互函数 --------------------
def fetch_indicators(session: requests.Session, cid: str) -> List[Dict[str, Any]]:
    """
    调用 /new/queryIndicatorsByCid?cid={cid}
    返回指标列表（每项包含 _id 和 i_showname）
    """
    url = f"{BASE_URL}/new/queryIndicatorsByCid?cid={cid}"
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        obj = resp.json()
    except Exception as e:
        logger.error("请求指标列表失败 cid=%s: %s", cid, e)
        return []

    data_candidates = []
    if isinstance(obj, dict):
        # 常见位置：顶层的 list/rows/indicators
        for key in ("list", "rows", "indicators"):
            val = obj.get(key)
            if isinstance(val, list):
                data_candidates = val
                break
        # 有些响应把实际列表放在 data 或 result 字段里，例如 data: { total:..., list: [...] }
        if not data_candidates and "data" in obj:
            data_obj = obj.get("data")
            if isinstance(data_obj, list):
                data_candidates = data_obj
            elif isinstance(data_obj, dict):
                if "list" in data_obj and isinstance(data_obj["list"], list):
                    data_candidates = data_obj["list"]
                elif "rows" in data_obj and isinstance(data_obj["rows"], list):
                    data_candidates = data_obj["rows"]
        if not data_candidates and "result" in obj:
            res = obj.get("result")
            if isinstance(res, list):
                data_candidates = res
            elif isinstance(res, dict):
                if "list" in res and isinstance(res["list"], list):
                    data_candidates = res["list"]
                elif "rows" in res and isinstance(res["rows"], list):
                    data_candidates = res["rows"]
    if not data_candidates and isinstance(obj, list):
        data_candidates = obj

    indicators = []
    for item in data_candidates:
        if not isinstance(item, dict):
            continue
        _id = item.get("_id") or item.get("id") or item.get("indicatorId")
        name = item.get("i_showname") or item.get("iShowname") or item.get("name") or item.get("showName")
        if _id and name:
            indicators.append({"_id": _id, "i_showname": name})
    if not indicators and isinstance(obj, dict) and "_id" in obj and "i_showname" in obj:
        indicators.append({"_id": obj.get("_id"), "i_showname": obj.get("i_showname")})
    logger.info("cid=%s 找到 %d 个指标", cid, len(indicators))
    return indicators


def parse_data_response(obj: Any, id2name: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    解析 /getEsDataByCidAndDt 返回体，返回统一的行结构列表：
    {'指标ID':..., '指标名称':..., '年份':..., '数值':..., '单位':...}
    """
    rows = []
    if not obj:
        return rows

    data_arr = None
    if isinstance(obj, dict):
        for key in ("data", "result", "rows"):
            if key in obj and isinstance(obj[key], list):
                data_arr = obj[key]
                break
    if data_arr is None and isinstance(obj, list):
        data_arr = obj

    if not data_arr:
        return rows

    for year_item in data_arr:
        if not isinstance(year_item, dict):
            continue
        code = year_item.get("code") or year_item.get("dt") or year_item.get("year")
        if not code:
            continue
        year = code
        if isinstance(year, str) and year.endswith("YY"):
            year = year[:-2]
        values = year_item.get("values") or year_item.get("value") or year_item.get("data") or []
        if not isinstance(values, list):
            continue
        for v in values:
            if not isinstance(v, dict):
                continue
            indicator_id = v.get("_id") or v.get("id")
            indicator_name = v.get("i_showname") or v.get("iShowname") or v.get("name") or id2name.get(indicator_id, "")
            value = v.get("value")
            if value is None:
                for key in ("v", "val", "data", "valueStr"):
                    if key in v:
                        value = v.get(key)
                        break
            unit = v.get("du_name") or v.get("unit") or v.get("duName") or ""
            rows.append({
                "指标ID": indicator_id,
                "指标名称": indicator_name,
                "年份": year,
                "数值": value,
                "单位": unit,
            })
    return rows


def fetch_data_by_cid_and_dt(session: requests.Session, cid: str, indicator_ids: List[str], year_range: str, area_code: str, id2name: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    调用 /stream/esData（POST）并解析
    """
    url = f"{BASE_URL}/stream/esData"
    # 根据 area_code 推断地区名称（12位代码 -> 截取前6位匹配 PROVINCE_MAP，兜底为"全国"）
    area_name = "全国"
    if area_code and area_code != "000000000000":
        prefix6 = area_code[:6] if len(area_code) >= 6 else area_code
        area_name = PROVINCE_MAP.get(prefix6, area_code)
    payload = {
        "cid": cid,
        "indicatorIds": indicator_ids,
        "daCatalogId": "",
        "das": [{"text": area_name, "value": area_code}],
        "dts": [year_range],
        "showType": SHOW_TYPE,
        "rootId": ROOT_ID,
    }
    try:
        resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        obj = resp.json()
    except Exception as e:
        logger.error("请求数据失败 cid=%s indicators=%d: %s", cid, len(indicator_ids), e)
        return []

    return parse_data_response(obj, id2name)


# -------------------- 主流程 --------------------
def process_dataset(session: requests.Session, dataset: Dict[str, Any]) -> pd.DataFrame:
    """
    处理单个数据集，支持全国与分省（region_type），返回 DataFrame（列：数据集名称、指标名称、地区、年份、数值、单位）
    """
    cid = dataset.get("cid")
    dataset_name = dataset.get("name", cid or "unknown")
    year_range = dataset.get("year_range", DEFAULT_YEAR_RANGE)
    region_type = dataset.get("region_type", "nationwide")

    logger.info("处理数据集：%s (cid=%s) 年份=%s 区域=%s", dataset_name, cid, year_range, region_type)

    indicators = fetch_indicators(session, cid)
    if not indicators:
        logger.warning("数据集 %s 未找到指标，跳过。", dataset_name)
        return pd.DataFrame(columns=["数据集名称", "指标名称", "地区", "年份", "数值", "单位"])

    indicator_ids = [it["_id"] for it in indicators if it.get("_id")]
    id2name = {it["_id"]: it.get("i_showname", "") for it in indicators if it.get("_id")}

    all_rows = []

    # 全国数据
    if region_type == "nationwide":
        area_code = normalize_area_code("000000")
        batches = list(split_batches(indicator_ids, INDICATOR_BATCH_SIZE))
        batch_iter = batches
        if tqdm:
            batch_iter = tqdm(batches, desc=f"Batches for {dataset_name}")
        for batch in batch_iter:
            try:
                resp_rows = fetch_data_by_cid_and_dt(session, cid, batch, year_range, area_code, id2name)
                for r in resp_rows:
                    if not r.get("指标名称"):
                        r["指标名称"] = id2name.get(r.get("指标ID"), "")
                    r_row = {
                        "数据集名称": dataset_name,
                        "指标名称": r.get("指标名称", ""),
                        "地区": "全国",
                        "年份": r.get("年份", ""),
                        "数值": r.get("数值"),
                        "单位": r.get("单位", ""),
                    }
                    all_rows.append(r_row)
            except Exception as e:
                logger.exception("处理 cid=%s 全国批次失败: %s", cid, e)
            try_sleep(1.0, 2.0)

    # 分省数据
    elif region_type == "provincial":
        prov_items = list(PROVINCE_MAP.items())
        prov_iter = prov_items
        if tqdm:
            prov_iter = tqdm(prov_items, desc=f"Provinces for {dataset_name}")
        for prov_code, prov_name in prov_iter:
            area_code = normalize_area_code(prov_code)
            batches = list(split_batches(indicator_ids, INDICATOR_BATCH_SIZE))
            batch_iter = batches
            if tqdm:
                batch_iter = tqdm(batches, desc=f"Batches [{prov_name}]", leave=False)
            for batch in batch_iter:
                try:
                    resp_rows = fetch_data_by_cid_and_dt(session, cid, batch, year_range, area_code, id2name)
                    for r in resp_rows:
                        if not r.get("指标名称"):
                            r["指标名称"] = id2name.get(r.get("指标ID"), "")
                        r_row = {
                            "数据集名称": dataset_name,
                            "指标名称": r.get("指标名称", ""),
                            "地区": prov_name,
                            "年份": r.get("年份", ""),
                            "数值": r.get("数值"),
                            "单位": r.get("单位", ""),
                        }
                        all_rows.append(r_row)
                except Exception as e:
                    logger.exception("处理 cid=%s 省份 %s 批次失败: %s", cid, prov_name, e)
                try_sleep(1.0, 2.0)
            # 省与省之间稍微等待
            try_sleep(0.5, 1.0)

    else:
        logger.warning("未知 region_type=%s，按全国处理。", region_type)
        return process_dataset(session, {**dataset, "region_type": "nationwide"})

    df = pd.DataFrame(all_rows, columns=["数据集名称", "指标名称", "地区", "年份", "数值", "单位"])

    # 保存为单独文件，按用户要求的命名格式：{数据集名称}_分省_{年份范围}.csv
    suffix_str = "分省" if region_type == "provincial" else ("全国" if region_type == "nationwide" else "other")
    safe_name = sanitize_filename(f"{dataset_name}_{suffix_str}_{year_range}")
    file_path = OUTPUT_DIR / f"{safe_name}.csv"
    try:
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        logger.info("已保存：%s （%d 行）", file_path, len(df))
    except Exception:
        logger.exception("保存文件失败：%s", file_path)

    return df


def main():
    session = create_session_with_retries()
    all_dfs = []
    # 支持命令行参数：仅分省抓取与续跑（跳过已存在文件）
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--provincial-only", action="store_true",
                        help="只处理 region_type == 'provincial' 的数据集")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已存在的输出文件（按文件名判断）")
    args = parser.parse_args()

    # 根据参数构建待处理数据集列表
    if args.provincial_only:
        dataset_list = [ds for ds in DATASETS if ds.get("region_type") == "provincial"]
    else:
        dataset_list = list(DATASETS)

    if args.resume:
        filtered = []
        for ds in dataset_list:
            dataset_name = ds.get("name", ds.get("cid", "unknown"))
            year_range = ds.get("year_range", DEFAULT_YEAR_RANGE)
            region_type = ds.get("region_type", "nationwide")
            suffix_str = "分省" if region_type == "provincial" else ("全国" if region_type == "nationwide" else "other")
            safe_name = sanitize_filename(f"{dataset_name}_{suffix_str}_{year_range}")
            file_path = OUTPUT_DIR / f"{safe_name}.csv"
            if file_path.exists():
                logger.info("跳过已存在文件：%s", file_path)
                continue
            filtered.append(ds)
        dataset_list = filtered

    dataset_iter = dataset_list
    if tqdm:
        dataset_iter = tqdm(dataset_list, desc="Datasets")

    for ds in dataset_iter:
        try:
            df = process_dataset(session, ds)
            if df is None or df.empty:
                logger.info("数据集 %s 没有数据，跳过保存。", ds.get("name"))
                continue
            all_dfs.append(df)
            if SAVE_PER_DATASET:
                region_type = ds.get("region_type", "nationwide")
                suffix_str = "分省" if region_type == "provincial" else ("全国" if region_type == "nationwide" else "other")
                safe_name = sanitize_filename(f"{ds.get('name')}_{suffix_str}_{ds.get('year_range', DEFAULT_YEAR_RANGE)}")
                file_path = OUTPUT_DIR / f"{safe_name}.csv"
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info("已保存：%s", file_path)
        except Exception as e:
            logger.exception("处理数据集 %s 时出现错误: %s", ds.get("name"), e)
        try_sleep(1.0, 2.0)

    if not all_dfs:
        logger.warning("未收集到任何数据。")
        return

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    try:
        combined.sort_values(by=["数据集名称", "指标名称", "年份"], inplace=True, ignore_index=True)
    except Exception:
        pass

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"agri_indicators_all_{ts}.csv"
    combined.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info("全部数据已合并并保存到：%s", out_file)
    logger.info("完成。")


if __name__ == "__main__":
    main()
