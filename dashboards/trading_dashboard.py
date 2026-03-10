import streamlit as st
import pandas as pd


class TradingDashboard:

    def run(self, df):

        st.title("AI Quant Trading Dashboard")

        st.line_chart(df["Close"])

        if "strategy_score" in df.columns:
            st.line_chart(df["strategy_score"])

        if "portfolio_value" in df.columns:
            st.line_chart(df["portfolio_value"])

        if "prediction_confidence" in df.columns:
            st.line_chart(df["prediction_confidence"])

        st.dataframe(df.tail())
