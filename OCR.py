                        # 使用 column_config 和 format 强制限制小数位数和列宽
                        st.dataframe(
                            df.style.apply(highlight_risk, axis=1).format({
                                "名义价值 (USD)": "${:,.2f}", 
                                "已收权利金": "${:,.2f}",
                                "安全垫 (%)": "{:.2f}",      # 强制格式化为 2 位小数
                                "期权浮盈 (%)": "{:.2f}"     # 强制格式化为 2 位小数
                            }),
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "标记": st.column_config.TextColumn("标记 🎯", width="large"),
                                "状态": st.column_config.TextColumn("状态", width="medium"),
                                "安全垫 (%)": st.column_config.NumberColumn("安全垫 (%)", format="%.2f"),
                                "期权浮盈 (%)": st.column_config.NumberColumn("期权浮盈 (%)", format="%.2f")
                            }
                        )

                        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("📥 一键下载分析数据为 CSV", data=csv_data, file_name="extracted_options.csv", mime="text/csv", use_container_width=True)
