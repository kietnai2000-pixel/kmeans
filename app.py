import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán giá nhà bằng K-Means", page_layout="wide"
)

st.title("🏡 Ứng dụng Dự đoán Giá nhà & Phân cụm K-Means")
st.markdown(
    "Mô hình học máy phân cụm dữ liệu bất động sản và dự đoán giá nhà."
)

# ---------------------------------------------------------
# Sidebar: Load dữ liệu và tham số
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình dữ liệu & Tham số")

train_file = st.sidebar.file_uploader(
    "Tải lên file Train (Excel)", type=["xlsx"]
)
test_file = st.sidebar.file_uploader("Tải lên file Test (Excel)", type=["xlsx"])


@st.cache_data
def load_data(uploaded_train, uploaded_test):
    if uploaded_train is not None:
        df_train = pd.read_excel(uploaded_train)
    elif os.path.exists("nhom4_train_data.xlsx"):
        df_train = pd.read_excel("nhom4_train_data.xlsx")
    elif os.path.exists("nhom4_train_data_2.xlsx"):
        df_train = pd.read_excel("nhom4_train_data_2.xlsx")
    else:
        df_train = None

    if uploaded_test is not None:
        df_test = pd.read_excel(uploaded_test)
    elif os.path.exists("nhom4_test_data.xlsx"):
        df_test = pd.read_excel("nhom4_test_data.xlsx")
    elif os.path.exists("nhom4_test_data_2.xlsx"):
        df_test = pd.read_excel("nhom4_test_data_2.xlsx")
    else:
        df_test = None

    return df_train, df_test


train_df, test_df = load_data(train_file, test_file)

if train_df is None or test_df is None:
    st.error(
        "Vui lòng tải lên file dữ liệu Train và Test hoặc đặt file `nhom4_train_data.xlsx` và `nhom4_test_data.xlsx` cùng thư mục code!"
    )
    st.stop()

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "HouseAge",
]
target = "SalePrice"

k_clusters = st.sidebar.slider("Chọn số lượng cụm (K)", 2, 10, 3)

# ---------------------------------------------------------
# Khởi tạo & Huấn luyện thuật toán K-Means từ đầu
# ---------------------------------------------------------
X_train = train_df[features].values
y_train = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

# Fit K-Means
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
train_df["Cluster"] = kmeans.fit_predict(X_train)
centroids = kmeans.cluster_centers_

# Tính giá trị trung bình SalePrice của từng cụm trên Train
cluster_price_map = train_df.groupby("Cluster")[target].mean().to_dict()

# Dự đoán trên tập Test:
# 1. Tính khoảng cách tới centroid gần nhất
test_df["Cluster"] = kmeans.predict(X_test)
# 2. Gán giá dự đoán bằng giá trung bình của cụm đó
test_df["Predicted_SalePrice"] = test_df["Cluster"].map(cluster_price_map)

# ---------------------------------------------------------
# Giao diện chính tab
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Dữ liệu & Phân cụm",
        "📈 Biểu đồ Elbow & Cụm",
        "📊 Dự đoán & Đánh giá Test",
        "🔮 Dự đoán nhà mới",
    ]
)

# TAB 1: DỮ LIỆU & PHÂN CỤM
with tab1:
    st.subheader("1. Xem trước dữ liệu gốc")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Tập huấn luyện (Train): {train_df.shape[0]} dòng**")
        st.dataframe(train_df.head(10))
    with col2:
        st.write(f"**Tập kiểm thử (Test): {test_df.shape[0]} dòng**")
        st.dataframe(test_df.head(10))

    st.subheader("2. Tâm các cụm (Centroids) & Đặc điểm trung bình")
    centroid_df = pd.DataFrame(centroids, columns=features)
    centroid_df["SalePrice_Mean"] = [
        cluster_price_map[i] for i in range(k_clusters)
    ]
    centroid_df.index.name = "Cluster"
    st.dataframe(
        centroid_df.style.highlight_max(axis=0, color="lightgreen")
    )

    st.markdown("### 💡 Giải thích ý nghĩa các cụm (K=3):")
    st.markdown(
        """
    - **Cụm 0 (Nhà mới, chất lượng cao):** Chỉ số `OverallQual` cao, `HouseAge` rất thấp (âm - nhà mới xây), giá trung bình cao.
    - **Cụm 1 (Nhà bình dân):** Chỉ số `OverallQual` âm (dưới trung bình), diện tích hầm và nhà xe ở mức vừa phải.
    - **Cụm 2 (Nhà lâu năm, Garage rộng):** `OverallQual` tốt, `GarageCars` cao nhưng `HouseAge` lớn (nhà đã xây dựng lâu).
    """
    )

# TAB 2: BIỂU ĐỒ ELBOW VÀ PHÂN CỤM
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Biểu đồ Elbow xác định K tối ưu")
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_train)
            wcss.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(K_range, wcss, "bo-", linewidth=2, markersize=8)
        ax_elbow.axvline(
            x=3, color="red", linestyle="--", label="Điểm Elbow đề xuất (K=3)"
        )
        ax_elbow.set_xlabel("Số lượng cụm K")
        ax_elbow.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
        ax_elbow.set_title("Phương pháp Elbow")
        ax_elbow.legend()
        ax_elbow.grid(True)
        st.pyplot(fig_elbow)

    with col_b:
        st.subheader("Biểu đồ phân bố giá nhà (SalePrice)")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.histplot(
            train_df[target], kde=True, ax=ax_dist, color="skyblue"
        )
        ax_dist.set_title("Phân bố biến mục tiêu SalePrice trên tập Train")
        ax_dist.set_xlabel("Giá nhà (SalePrice)")
        ax_dist.set_ylabel("Tần suất")
        st.pyplot(fig_dist)

    st.subheader("Biểu đồ phân cụm (Giảm chiều dữ liệu PCA 2D)")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    centroids_pca = pca.transform(centroids)

    fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
    scatter = ax_pca.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=train_df["Cluster"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    ax_pca.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        linewidths=2,
        label="Tâm cụm (Centroids)",
    )
    ax_pca.set_title("Trực quan hóa các cụm nhà và Tâm cụm trong không gian 2D")
    ax_pca.set_xlabel("Thành phần chính 1 (PCA 1)")
    ax_pca.set_ylabel("Thành phần chính 2 (PCA 2)")
    ax_pca.legend()
    ax_pca.grid(True)
    st.pyplot(fig_pca)

# TAB 3: DỰ ĐOÁN & ĐÁNH GIÁ TRÊN TẬP TEST
with tab3:
    st.subheader("1. Các chỉ số đánh giá trên tập Kiểm thử (Test)")

    mae = mean_absolute_error(test_df[target], test_df["Predicted_SalePrice"])
    rmse = np.sqrt(
        mean_squared_error(test_df[target], test_df["Predicted_SalePrice"])
    )
    r2 = r2_score(test_df[target], test_df["Predicted_SalePrice"])

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (Sai số tuyệt đối TB)", f"${mae:,.2f}")
    m2.metric("RMSE (Căn sai số bình phương TB)", f"${rmse:,.2f}")
    m3.metric("R² Score (Hệ số xác định)", f"{r2:.4f}")

    st.subheader(
        "2. Bảng kết quả dự đoán chi tiết đối chiếu trên tập Test (nhom4_test_data.xlsx)"
    )
    test_display = test_df[
        features
        + [
            target,
            "Cluster",
            "Predicted_SalePrice",
        ]
    ].copy()
    test_display["Absolute_Error"] = np.abs(
        test_display[target] - test_display["Predicted_SalePrice"]
    )
    st.dataframe(test_display)

    st.subheader("3. Biểu đồ so sánh Giá thực tế và Giá ước tính")
    fig_comp, ax_comp = plt.subplots(figsize=(9, 4.5))
    ax_comp.plot(
        range(len(test_df)),
        test_df[target],
        label="Giá thực tế (Actual)",
        color="blue",
        alpha=0.7,
    )
    ax_comp.plot(
        range(len(test_df)),
        test_df["Predicted_SalePrice"],
        label="Giá dự đoán (Predicted)",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    ax_comp.set_title(
        "So sánh Giá nhà Thực tế vs Giá nhà Dự đoán theo thuật toán K-Means"
    )
    ax_comp.set_xlabel("Chỉ số mẫu nhà trong tập Test")
    ax_comp.set_ylabel("Giá nhà (SalePrice)")
    ax_comp.legend()
    ax_comp.grid(True)
    st.pyplot(fig_comp)

# TAB 4: DỰ ĐOÁN CHO NHÀ MỚI NGHỆ THUẬT
with tab4:
    st.subheader("Dự đoán giá cho một ngôi nhà mới bất kỳ")
    c1, c2, c3 = st.columns(3)
    in_qual = c1.number_input("OverallQual (Chất lượng)", value=0.0, step=0.1)
    in_area = c2.number_input("GrLivArea (Diện tích sống)", value=0.0, step=0.1)
    in_cars = c3.number_input(
        "GarageCars (Sức chứa Garage)", value=0.0, step=0.1
    )

    c4, c5 = st.columns(2)
    in_bsmt = c4.number_input("TotalBsmtSF (Diện tích hầm)", value=0.0, step=0.1)
    in_age = c5.number_input("HouseAge (Tuổi thọ nhà)", value=0.0, step=0.1)

    if st.button("Dự đoán giá nhà ngay"):
        sample = np.array([[in_qual, in_area, in_cars, in_bsmt, in_age]])
        pred_cluster = kmeans.predict(sample)[0]
        pred_price = cluster_price_map[pred_cluster]

        st.success(f"🏠 Ngôi nhà thuộc **Cụm {pred_cluster}**")
        st.info(f"💵 Giá dự đoán ước tính: **${pred_price:,.2f}**")
