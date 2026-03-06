import streamlit as st
import psutil


class MonitoringDashboard:

    def run(self):

        st.title("System Monitoring")

        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        st.metric("CPU Usage", cpu)
        st.metric("Memory Usage", memory)