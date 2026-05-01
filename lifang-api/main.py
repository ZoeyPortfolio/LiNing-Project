import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
import io
import tempfile
import shutil

warnings.filterwarnings('ignore')

app = FastAPI(title="李宁快闪店分析API")

train_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
target_cols = ['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']


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


def calculate_series_ratio(sales_detail_df):
    series_mapping = {
        '李宁荣耀金标': '金标',
        '李宁荣耀': '荣耀',
        '国家队': '国家队',
        '其他系列': '其他'
    }
    sales_detail_df['系列'] = sales_detail_df['系列'].map(series_mapping)
    
    sales_detail_filtered = sales_detail_df[sales_detail_df['品类'] != '推广类'].copy()
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
    return category_ratio_wide


@app.post("/analyze")
async def analyze(
    population_file: UploadFile = File(...),
    phone_file: UploadFile = File(...),
    age_file: UploadFile = File(...),
    gender_file: UploadFile = File(...),
    asset_file: UploadFile = File(...),
    sales_file: UploadFile = File(...),
    flash_mapping_file: UploadFile = File(...),
    n_clusters: int = Form(3)
):
    temp_dir = tempfile.mkdtemp()
    
    try:
        population_df = pd.read_excel(io.BytesIO(await population_file.read()))
        phone_df = pd.read_excel(io.BytesIO(await phone_file.read()))
        age_df = pd.read_excel(io.BytesIO(await age_file.read()))
        gender_df = pd.read_excel(io.BytesIO(await gender_file.read()))
        asset_df = pd.read_excel(io.BytesIO(await asset_file.read()))
        sales_detail = pd.read_excel(io.BytesIO(await sales_file.read()))
        flash_mapping = pd.read_excel(io.BytesIO(await flash_mapping_file.read()))
        
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        flash_mapping = flash_mapping[['店铺名称', '李宁商场名称']].drop_duplicates()
        all_features_df = df[['李宁商场名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']].copy()
        liNing_store_names = flash_mapping['李宁商场名称'].unique().tolist()
        matched_features = all_features_df[all_features_df['李宁商场名称'].isin(liNing_store_names)].copy()
        
        cluster_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
        scaler_flash = StandardScaler()
        X_scaled = scaler_flash.fit_transform(matched_features[cluster_features])
        
        kmeans_flash = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        matched_features['客群类型'] = kmeans_flash.fit_predict(X_scaled)
        
        name_mapping = {0: "一线城市标杆店", 1: "女性高消潜力店", 2: "年轻潮流主力店"}
        matched_features['客群类型名称'] = matched_features['客群类型'].map(name_mapping)
        
        feature_mapping = matched_features[['李宁商场名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', '客群类型名称']].drop_duplicates()
        result_df = flash_mapping.merge(feature_mapping, on='李宁商场名称', how='left')
        
        category_ratio = calculate_series_ratio(sales_detail)
        final_df = category_ratio.merge(
            result_df[['店铺名称', '客群类型名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']],
            on='店铺名称', how='inner'
        )
        
        type_summary = final_df.groupby('客群类型名称')[target_cols].mean().round(4).to_dict()
        
        return JSONResponse(content={
            "status": "success",
            "flash_cluster_result": result_df[['店铺名称', '客群类型名称']].to_dict(orient='records'),
            "type_summary": type_summary
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)