import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
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


@app.post("/top20")
async def top20(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...),
    n_clusters: int = Form(3)
):
    try:
        # 下载文件
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


@app.post("/cluster-result")
async def cluster_result(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...),
    n_clusters: int = Form(3)
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
        
        scaler_all = StandardScaler()
        X_scaled_all = scaler_all.fit_transform(all_malls_df[cluster_features])
        kmeans_all = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        all_malls_df['客群类型'] = kmeans_all.fit_predict(X_scaled_all)
        
        name_mapping = {0: "一线城市标杆店", 1: "女性高消潜力店", 2: "年轻潮流主力店"}
        all_malls_df['客群类型名称'] = all_malls_df['客群类型'].map(name_mapping)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_malls_df.to_excel(writer, index=False, sheet_name='聚类结果')
        output.seek(0)
        
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=cluster_result.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature-means")
async def feature_means(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...),
    n_clusters: int = Form(3)
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
        
        scaler_all = StandardScaler()
        X_scaled_all = scaler_all.fit_transform(all_malls_df[cluster_features])
        kmeans_all = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        all_malls_df['客群类型'] = kmeans_all.fit_predict(X_scaled_all)
        
        name_mapping = {0: "一线城市标杆店", 1: "女性高消潜力店", 2: "年轻潮流主力店"}
        all_malls_df['客群类型名称'] = all_malls_df['客群类型'].map(name_mapping)
        
        mean_result = all_malls_df.groupby('客群类型名称')[cluster_features].mean().round(2)
        
        return JSONResponse(content=mean_result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series-ratio")
async def series_ratio(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...),
    n_clusters: int = Form(3)
):
    try:
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        sales_detail = download_file(sales_file)
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        series_mapping = {'李宁荣耀金标': '金标', '李宁荣耀': '荣耀', '国家队': '国家队', '其他系列': '其他'}
        sales_detail['系列'] = sales_detail['系列'].map(series_mapping)
        sales_detail_filtered = sales_detail[sales_detail['品类'] != '推广类'].copy()
        sales_detail_filtered = sales_detail_filtered[pd.notna(sales_detail_filtered['销售数量'])]
        sales_detail_filtered = sales_detail_filtered[sales_detail_filtered['销售数量'] > 0]
        
        summary = sales_detail_filtered.groupby(['店铺名称', '系列'])['销售数量'].sum().reset_index()
        total_by_store = summary.groupby('店铺名称')['销售数量'].sum().reset_index()
        total_by_store.columns = ['店铺名称', '总销售数量']
        summary = summary.merge(total_by_store, on='店铺名称')
        summary['销售占比'] = summary['销售数量'] / summary['总销售数量']
        
        category_ratio_wide = summary.pivot_table(index='店铺名称', columns='系列', values='销售占比', fill_value=0).reset_index()
        category_ratio_wide = category_ratio_wide.rename(columns={'金标': '金标Proportion', '荣耀': '荣耀Proportion', '国家队': '国家队Proportion', '其他': '其他Proportion'})
        
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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
