import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import io
import requests

warnings.filterwarnings('ignore')

app = FastAPI(title="李宁快闪店分析API")

train_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
target_cols = ['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']


def download_file(url: str) -> pd.DataFrame:
    """从URL下载Excel文件并返回DataFrame"""
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_excel(io.BytesIO(response.content))


def load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df):
    phone_columns = ['赢商项目ID'] + ['APPLE', 'HUAWEI', 'SAMSUNG']
    phone_df = phone_df[phone_columns]
    
    age_columns = ['赢商项目ID'] + ['19-24', '25-29']
    age_df = age_df[age_columns]
    
    gender_columns = ['赢商项目ID', '女性占比']
    gender_df = gender_df[gender_columns]
    
    asset_columns = ['赢商项目ID'] + ['超级富豪', '富豪', '中产']
    asset_df = asset_df[asset_columns]
    
    df = population_df.merge(phone_df, on='赢商项目ID', how='inner')
    df = df.merge(age_df, on='赢商项目ID', how='inner')
    df = df.merge(gender_df, on='赢商项目ID', how='inner')
    df = df.merge(asset_df, on='赢商项目ID', how='inner')
    
    df['年轻占比'] = df['19-24'] + df['25-29']
    df['高消费力_资产'] = df['超级富豪'] + df['富豪'] + df['中产']
    df['高消费力_手机'] = df['APPLE'] + df['HUAWEI'] + df['SAMSUNG']
    df['高消费力'] = (df['高消费力_资产'] + df['高消费力_手机']) / 2
    
    province_score = {
        '上海市': 4, '北京市': 4, '广东省': 4,
        '江苏省': 3, '浙江省': 3, '四川省': 3, '湖北省': 3, '湖南省': 3,
        '河南省': 3, '安徽省': 3, '福建省': 3, '陕西省': 3, '重庆市': 3,
        '天津市': 3, '山东省': 3, '辽宁省': 3,
        '河北省': 2, '江西省': 2, '广西壮族自治区': 2, '云南省': 2,
        '贵州省': 2, '山西省': 2, '吉林省': 2, '黑龙江省': 2,
    }
    df['省份分数'] = df['省份'].map(province_score)
    return df


# ==================== 固定聚类 ====================
def perform_clustering(df, cluster_features):
    """执行K-means聚类，固定3类，输出固定名称"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])
    
    # K=3 固定聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['客群类型'] = kmeans.fit_predict(X_scaled)
    
    # 根据聚类中心特征智能命名
    # 计算各聚类中心的均值，用于判断类型
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=cluster_features)
    
    # 判断逻辑：
    # 类型0：省份分数高 + 工作人口多 → 一线城市标杆店
    # 类型1：女性占比高 → 女性高消潜力店
    # 类型2：年轻占比高 → 年轻潮流主力店
    type_names = {}
    
    for cluster_id in range(3):
        row = center_df.loc[cluster_id]
        # 找出每个聚类的主要特征
        if row['省份分数'] >= 3.5 and row['3公里工作人口'] > 300000:
            type_names[cluster_id] = "一线城市标杆店"
        elif row['女性占比'] > 0.5:
            type_names[cluster_id] = "女性高消潜力店"
        else:
            type_names[cluster_id] = "年轻潮流主力店"
    
    # 如果还有未分配的，按顺序补充
    assigned = set(type_names.values())
    all_names = ["一线城市标杆店", "女性高消潜力店", "年轻潮流主力店"]
    for name in all_names:
        if name not in assigned:
            for cluster_id in range(3):
                if cluster_id not in type_names:
                    type_names[cluster_id] = name
                    break
    
    df['客群类型名称'] = df['客群类型'].map(type_names)
    
    return df, center_df


# ==================== 端点1：聚类结果 ====================
@app.post("/cluster-result")
async def cluster_result(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    try:
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        cluster_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
        all_malls_df = df[['李宁商场名称', '城市', '省份'] + cluster_features].copy()
        all_malls_df = all_malls_df.dropna(subset=cluster_features)
        
        # 固定聚类
        all_malls_df, center_df = perform_clustering(all_malls_df, cluster_features)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_malls_df.to_excel(writer, index=False, sheet_name='聚类结果')
            center_df.to_excel(writer, index=True, sheet_name='聚类中心')
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=cluster_result.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 端点2：特征均值 ====================
@app.post("/feature-means")
async def feature_means(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    try:
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        cluster_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
        all_malls_df = df[['李宁商场名称'] + cluster_features].copy()
        all_malls_df = all_malls_df.dropna(subset=cluster_features)
        
        # 固定聚类
        all_malls_df, center_df = perform_clustering(all_malls_df, cluster_features)
        
        mean_result = all_malls_df.groupby('客群类型名称')[cluster_features].mean().round(4)
        
        return JSONResponse(content={
            "status": "success",
            "data": mean_result.to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 端点3：系列占比预测（简化版）====================
@app.post("/series-ratio")
async def series_ratio(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    try:
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        sales_detail = download_file(sales_file)
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        # 系列名称映射
        series_mapping = {
            '李宁荣耀金标': '金标',
            '李宁荣耀': '荣耀',
            '国家队': '国家队',
            '其他系列': '其他'
        }
        sales_detail['系列'] = sales_detail['系列'].map(series_mapping)
        
        sales_detail_filtered = sales_detail[sales_detail['品类'] != '推广类'].copy()
        sales_detail_filtered = sales_detail_filtered[pd.notna(sales_detail_filtered['销售数量'])]
        sales_detail_filtered = sales_detail_filtered[sales_detail_filtered['销售数量'] > 0]
        
        summary = sales_detail_filtered.groupby(['店铺名称', '系列'])['销售数量'].sum().reset_index()
        total_by_store = summary.groupby('店铺名称')['销售数量'].sum().reset_index()
        total_by_store.columns = ['店铺名称', '总销售数量']
        summary = summary.merge(total_by_store, on='店铺名称')
        summary['销售占比'] = summary['销售数量'] / summary['总销售数量']
        
        category_ratio_wide = summary.pivot_table(
            index='店铺名称', columns='系列', values='销售占比', fill_value=0
        ).reset_index()
        
        for s in ['金标', '荣耀', '国家队', '其他']:
            if s not in category_ratio_wide.columns:
                category_ratio_wide[s] = 0
        
        category_ratio_wide = category_ratio_wide.rename(columns={
            '金标': '金标Proportion',
            '荣耀': '荣耀Proportion',
            '国家队': '国家队Proportion',
            '其他': '其他Proportion'
        })
        
        ratio_cols = ['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        row_sum = category_ratio_wide[ratio_cols].sum(axis=1)
        for col in ratio_cols:
            category_ratio_wide[col] = category_ratio_wide[col] / row_sum
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            category_ratio_wide.to_excel(writer, index=False, sheet_name='系列占比')
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=series_ratio.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 端点4：TOP20 ====================
@app.post("/top20")
async def top20(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    try:
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        tier1_cities = ['上海市', '北京市', '深圳市', '广州市']
        new_tier1_cities = ['成都市', '杭州市', '重庆市', '武汉市', '苏州市', '西安市', '南京市', 
                             '长沙市', '郑州市', '天津市', '合肥市', '青岛市', '东莞市', '宁波市']
        
        df_filtered = df[df['城市'].isin(tier1_cities + new_tier1_cities)]
        top20_result = df_filtered.nlargest(20, '高消费力')[['李宁商场名称', '城市', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']]
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            top20_result.to_excel(writer, index=False, sheet_name='TOP20')
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=top20.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 健康检查 ====================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
