from datetime import date
import numpy as np
import pandas as pd
import pickle
import mysql.connector
import streamlit as st
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
import json as js
import requests
import warnings
warnings.filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Forecast', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and-*+ position
    st.markdown(f'<h1 style="text-align: center;">Retail Sales Forecast</h1>',
                unsafe_allow_html=True)
    add_vertical_space(1)

# def lottie file
def lottie(filepath):

    with open(filepath, 'r') as file:

        return js.load(file)
# custom style for submit button - color and width

def style_submit_button():

    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)


# custom style for prediction result text - color and position

def style_prediction():

    st.markdown(
        """
            <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
        unsafe_allow_html=True
    )


# SQL columns ditionary

def columns_dict():

    columns_dict = {'Day': 'Day', 'Month': 'Month', 'Year': 'Year', 'Store': 'Store',
                    'Dept': 'Dept', 'Type': 'Type', 'Weekly_Sales': 'Weekly_Sales',
                    'Size': 'Size', 'IsHoliday': 'IsHoliDay', 'Temperature': 'Temperature',
                    'Fuel_Price': 'Fuel_Price', 'MarkDown1': 'MarkDown1', 'MarkDown2': 'MarkDown2',
                    'MarkDown3': 'MarkDown3', 'MarkDown4': 'MarkDown4', 'MarkDown5': 'MarkDown5',
                    'CPI': 'CPI', 'Unemployment': 'Unemployment'}
    return columns_dict



class plotly:

    def pie_chart(df, x, y, title, title_x=0.20):

        fig = px.pie(df, names=x, values=y, hole=0.5, title=title)

        fig.update_layout(title_x=title_x, title_font_size=22)

        fig.update_traces(text=df[y], textinfo='percent+value',
                          textposition='outside',
                          textfont=dict(color='white'),
                          outsidetextfont=dict(size=14))

        st.plotly_chart(fig, use_container_width=True)


    def vertical_bar_chart(df, x, y, text, color, title, title_x=0.25):

        fig = px.bar(df, x=x, y=y, labels={x: '', y: ''}, title=title)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(title_x=title_x, title_font_size=22)

        df[y] = df[y].astype(float)
        text_position = ['inside' if val >= max(
            df[y]) * 0.90 else 'outside' for val in df[y]]

        fig.update_traces(marker_color=color,
                          text=df[text],
                          textposition=text_position,
                          texttemplate='%{y}',
                          textfont=dict(size=14),
                          insidetextfont=dict(color='white'),
                          textangle=0,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)


    def scatter_chart(df, x, y):

        fig = px.scatter(data_frame=df, x=x, y=y, size=y, color=y, 
                         labels={x: '', y: ''}, title=columns_dict()[x])
        
        fig.update_layout(title_x=0.4, title_font_size=22)
        
        fig.update_traces(hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}")
        
        st.plotly_chart(fig, use_container_width=True, height=100)



class sql:

    def create_table():

        try:

            mydb = mysql.connector.connect(host='localhost',
                                    user='root',
                                    password='vino8799',
                                    database='retail_forecast')
            cursor = mydb.cursor()

            cursor.execute(f'''create table if not exists sales(
                                    Day           	int,
                                    Month           int,
                                    Year            int,
                                    Store           int,
                                    Dept            int,
                                    Type            int,
                                    Weekly_Sales    float,
                                    Size            int,
                                    IsHoliday      int,
                                    Temperature     float,
                                    Fuel_Price      float,
                                    MarkDown1       float,
                                    MarkDown2       float,
                                    MarkDown3       float,
                                    MarkDown4       float,
                                    MarkDown5       float,
                                    CPI             float,
                                    Unemployment    float);''')

            mydb.commit()
            cursor.close()
            mydb.close()
        
        except:
            
            st.warning("There is no database named 'retail_forecast'. Please create the database.")


    def drop_table():

        try:

            mydb = mysql.connector.connect(host='localhost',
                                    user='root',
                                    password='vino8799',
                                    database='retail_forecast')
            cursor = mydb.cursor()

            cursor.execute(f'''drop table if exists sales;''')

            mydb.commit()
            cursor.close()
            mydb.close()

        except:
            pass
    

    def data_migration():
        
        try:
            f = pd.read_csv(r'V:\project\vk_project\retail_sales\df_sql.csv')
            df = pd.DataFrame(f)

            mydb = mysql.connector.connect(host="localhost",
                                           user="root",
                                           password="vino8799",
                                           database='retail_forecast',
                                           connection_timeout=600)  # Increased timeout
            cursor = mydb.cursor()

            # Insert data in chunks
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                cursor.executemany('''INSERT INTO sales(Day, Month, Year, Store, Dept, Type,
                                                        Weekly_Sales, Size, IsHoliday,
                                                        Temperature, Fuel_Price, MarkDown1,
                                                        MarkDown2, MarkDown3, MarkDown4,
                                                        MarkDown5, CPI, Unemployment) 
                                      VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                             %s, %s, %s);''', chunk.values.tolist())
                mydb.commit()

            cursor.close()
            mydb.close()

        except mysql.connector.Error as err:
            print(f"Error: {err}")
        except Exception as e:
            print(f"Error: {e}")



class top_sales:

    # sales table order by date,Store and Dept

    def sql(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select * from sales
                            where {condition}
                            order by Year, Month, Day, Store, Dept asc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()

        return df


    # Year list from sales table 
    
    def Year():
        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Year from sales
                           order by Year asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Month list from sales table based on selected Year

    def Month(Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Month from sales
                           where Year='{Year}'
                           order by Month asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Day list from sales table based on selected Year, Month

    def Day(Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Day from sales
                           where Year='{Year}' and Month='{Month}'
                           order by Day asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Store list from sales table based on selected Year,Month, Day

    def Store(Day, Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store from sales
                           where Day='{Day}' and Year='{Year}' and Month='{Month}'
                           order by Store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # department list from sales table based on selected Year,Month, Day, Store
    
    def Dept(Day, Month, Year, Store):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Dept from sales
                           where Day='{Day}' and Month='{Month}' and Year='{Year}' and Store='{Store}'
                           order by Dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # department list (with 'Overall') from sales table based on selected Year,Month, Day

    def top_Store_Dept(Day, Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Dept from sales
                           where Day='{Day}' and Month='{Month}' and Year='{Year}' 
                           order by Dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]
        data.insert(0, 'Overall')

        cursor.close()
        mydb.close()

        return data


    # top 10 Stores filter options

    def top_Store_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            Year = st.selectbox(label='Year ', options=top_sales.Year())

        with col2:
            Month = st.selectbox(label='Month ', options=top_sales.Month(Year))

        with col1:
            Day = st.selectbox(
                label='Day ', options=top_sales.Day(Month, Year))

        with col4:
            Dept = st.selectbox(
                label='Dept ', options=top_sales.top_Store_Dept(Day, Month, Year))

        return Day, Month, Year, Dept


    # top 10 Stores based on Weekly_Sales

    def top_Store_sales(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select Store, sum(Weekly_Sales) as Weekly_Sales
                            from sales
                            where {condition}
                            group by Store
                            order by Weekly_Sales desc
                            limit 10;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['Store', 'Weekly Sales'], index=index)
        df['Weekly Sales'] = df['Weekly Sales'].apply(lambda x: f"{x:.2f}")
        df['Store_x'] = df['Store'].apply(lambda x: str(x)+'*')
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()

        return df


    # Store list (with 'Overall') from sales table

    def top_Dept_Store(Day, Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store from sales
                           where Day='{Day}' and Month='{Month}' and Year='{Year}' 
                           order by Store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]
        data.insert(0, 'Overall')

        cursor.close()
        mydb.close()

        return data


    # top 10 departments filter options

    def top_Dept_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            Year = st.selectbox(label='Year  ', options=top_sales.Year())

        with col2:
            Month = st.selectbox(
                label='Month  ', options=top_sales.Month(Year))

        with col1:
            Day = st.selectbox(
                label='Day  ', options=top_sales.Day(Month, Year))

        with col4:
            Store = st.selectbox(
                label='Store  ', options=top_sales.top_Dept_Store(Day, Month, Year))

        return Day, Month, Year, Store


    # top 10 departments based on Weekly_Sales

    def top_Dept_sales(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select Dept, sum(Weekly_Sales) as Weekly_Sales
                            from sales
                            where {condition}
                            group by Dept
                            order by Weekly_Sales desc
                            limit 10;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['Dept', 'Weekly Sales'], index=index)
        df['Weekly Sales'] = df['Weekly Sales'].apply(lambda x: f"{x:.2f}")
        df['Dept_x'] = df['Dept'].apply(lambda x: str(x)+'*')
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()

        return df



class comparison:

    # sales table order by date,Store,Dept

    def sql(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select * from sales
                            where {condition}
                            order by Year, Month, Day, Store, Dept asc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()

        return df


    # vertical line for st.metrics()

    def vertical_line():
        line_width = 2
        line_color = 'grey'

        # Use HTML and CSS to create the vertical line
        st.markdown(
            f"""
            <style>
                .vertical-line {{
                    border-left: {line_width}px solid {line_color};
                    height: 100vh;
                    position: absolute;
                    left: 55%;
                    margin-left: -{line_width / 2}px;
                }}
            </style>
            <div class="vertical-line"></div>
            """,
            unsafe_allow_html=True
        )


    # Year list from sales table

    def Year():
        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Year from sales
                           order by Year asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Month list from sales table based on Year

    def Month(Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Month from sales
                           where Year='{Year}'
                           order by Month asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Day list from sales table based on Year,Month

    def Day(Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Day from sales
                           where Year='{Year}' and Month='{Month}'
                           order by Day asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Store list from sales table based on Year,Month,Day

    def Store(Day, Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store from sales
                           where Day='{Day}' and Year='{Year}' and Month='{Month}'
                           order by Store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # department list from sales table based on Year,Month,Day,Store

    def Dept(Day, Month, Year, Store):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Dept from sales
                           where Day='{Day}' and Month='{Month}' and Year='{Year}' and Store='{Store}'
                           order by Dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Day list from sales table based on Year,Month

    def previous_week_Day(Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Day from sales
                           where Year='{Year}' and Month='{Month}'
                           order by Day asc;''')

        s = cursor.fetchall()

        if Month == 2 and Year == 2010:
            data = [i[0] for i in s]
            data.remove(data[0])
        else:
            data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # previous week filter options

    def previous_week_filter_options():

        col1, col2, col3, col4, col5 = st.columns(5, gap='medium')

        with col3:
            Year = st.selectbox(label='Year', options=comparison.Year())

        with col2:
            Month = st.selectbox(label='Month', options=comparison.Month(Year))

        with col1:
            Day = st.selectbox(
                label='Day', options=comparison.previous_week_Day(Month, Year))

        with col4:
            Store = st.selectbox(
                label='Store', options=comparison.Store(Day, Month, Year))

        with col5:
            Dept = st.selectbox(
                label='Dept', options=comparison.Dept(Day, Month, Year, Store))

        return Day, Month, Year, Store, Dept


    # comparison between current week and previous week

    def previous_week_sales_comparison(df, Day, Month, Year):

        index = df.index[(df['Day'] == Day) & (df['Month'] == Month) & (df['Year'] == Year)]
        current_index = index[0]-1
        previous_index = index[0]-2

        previous_data, current_data = {}, {}
        column_names = df.columns
        for i in range(0, len(column_names)):
            current_data[column_names[i]] = df.iloc[current_index, i]
            previous_data[column_names[i]] = df.iloc[previous_index, i]

        previous_date = f"{previous_data['Day']}-{previous_data['Month']}-{previous_data['Year']}"

        holiDay = {0: 'No', 1: 'Yes'}
        Type = {1: 'A', 2: 'B', 3: 'C'}
        st.code(f'''Type : {Type[current_data['Type']]}        Size : {current_data['Size']}        HoliDay : Previous Week = {holiDay[previous_data['IsHoliday']]} ({previous_date}) ;     Current Week = {holiDay[current_data['IsHoliday']]}''')

        comparison.vertical_line()
        col1, col2 = st.columns([0.55, 0.45])

        with col1:
            for i in ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                p, c = previous_data[i], current_data[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col3, col4, col5, col6 = st.columns([0.3, 0.4, 0.25, 0.05])
                    with col3:
                        add_vertical_space(1)
                        st.markdown(f"#### {previous_data[i]:.2f}")

                    with col4:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col5:
                        add_vertical_space(1)
                        st.markdown(f"#### {current_data[i]:.2f}")

        with col2:
            for i in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
                p, c = previous_data[i], current_data[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col7, col8, col9, col10 = st.columns(
                        [0.05, 0.3, 0.4, 0.25])
                    with col8:
                        add_vertical_space(1)
                        st.markdown(f"#### {previous_data[i]:.2f}")

                    with col9:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col10:
                        add_vertical_space(1)
                        st.markdown(f"#### {current_data[i]:.2f}")


    # manual tab filter options

    def manual_filter_options():

        col16, col17 = st.columns(2, gap='large')

        with col16:

            col6, col7, col8 = st.columns(3)

            with col8:
                Year1 = st.selectbox(label='Year1', options=comparison.Year())

            with col7:
                Month1 = st.selectbox(
                    label='Month1', options=comparison.Month(Year1))

            with col6:
                Day1 = st.selectbox(
                    label='Day1', options=comparison.Day(Month1, Year1))

            col9A, col9, col10, col10A = st.columns([0.1, 0.4, 0.4, 0.1])

            with col9:
                Store1 = st.selectbox(
                    label='Store1', options=comparison.Store(Day1, Month1, Year1))

            with col10:
                Dept1 = st.selectbox(label='Dept1', options=comparison.Dept(
                    Day1, Month1, Year1, Store1))

        with col17:

            col11, col12, col13 = st.columns(3)

            with col13:
                Year2 = st.selectbox(label='Year2', options=comparison.Year())

            with col12:
                Month2 = st.selectbox(
                    label='Month2', options=comparison.Month(Year2))

            with col11:
                Day2 = st.selectbox(
                    label='Day2', options=comparison.Day(Month2, Year2))

            col14A, col14, col15, col15A = st.columns([0.1, 0.4, 0.4, 0.1])

            with col14:
                manual_Store = comparison.Store(Day2, Month2, Year2)
                manual_Store[0], manual_Store[1] = manual_Store[1], manual_Store[0]
                Store2 = st.selectbox(label='Store2', options=manual_Store)

            with col15:
                if Year1 == Year2 and Month1 == Month2 and Day1 == Day2 and Store1 == Store2:
                    Dept = comparison.Dept(Day2, Month2, Year2, Store2)
                    Dept.remove(Dept1)
                    Dept2 = st.selectbox(label='Dept2', options=Dept)
                else:
                    Dept2 = st.selectbox(label='Dept2', options=comparison.Dept(
                        Day2, Month2, Year2, Store2))

        return Day1, Month1, Year1, Store1, Dept1, Day2, Month2, Year2, Store2, Dept2


    # comparison between 2 different Stores and department combination

    def manual_comparison(df1, df2):

        data1 = df1.iloc[0, :]
        df1_dict = data1.to_dict()

        data2 = df2.iloc[0, :]
        df2_dict = data2.to_dict()

        col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
        with col2:
            holiDay = {0: 'No', 1: 'Yes'}
            Type = {1: 'A', 2: 'B', 3: 'C'}
            st.code(f'''{Type[df1_dict['Type']]} : Type : {Type[df2_dict['Type']]}           {df1_dict['Size']} : Size : {df2_dict['Size']}           {holiDay[df1_dict['IsHoliday']]}  :  HoliDay : {holiDay[df2_dict['IsHoliday']]}''')

        comparison.vertical_line()
        col1, col2 = st.columns([0.55, 0.45])

        with col1:

            for i in ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                p, c = df1_dict[i], df2_dict[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col3, col4, col5, col6 = st.columns([0.3, 0.4, 0.25, 0.05])
                    with col3:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col4:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col5:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")

        with col2:

            for i in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
                p, c = df1_dict[i], df2_dict[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col7, col8, col9, col10 = st.columns(
                        [0.05, 0.3, 0.4, 0.25])
                    with col8:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col9:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col10:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")


    # compare between selected Store and top 10 Stores - filter options

    def top_Store_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            Year = st.selectbox(label='Year ', options=comparison.Year())

        with col2:
            Month = st.selectbox(
                label='Month ', options=comparison.Month(Year))

        with col1:
            Day = st.selectbox(
                label='Day ', options=comparison.Day(Month, Year))

        with col4:
            Store = st.selectbox(
                label='Store ', options=comparison.Store(Day, Month, Year))

        return Day, Month, Year, Store


    # Store wise Weekly_Sales and remaining columns avg from sales table

    def top_Store_sales(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store, Type, sum(Weekly_Sales) as Weekly_Sales,  
                        Size, IsHoliday, avg(Temperature) as Temperature,  
                        avg(Fuel_Price) as Fuel_Price, avg(MarkDown1) as MarkDown1,  
                        avg(MarkDown2) as MarkDown2, avg(MarkDown3) as MarkDown3,  
                        avg(MarkDown4) as MarkDown4, avg(MarkDown5) as MarkDown5, 
                        avg(CPI) as CPI, avg(Unemployment) as Unemployment
                       
                        from sales
                        where {condition}
                        group by Store, Type, Size, IsHoliday               
                        order by Weekly_Sales desc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]
        df = pd.DataFrame(s, columns=columns, index=index)
        df['Weekly_Sales'] = df['Weekly_Sales'].apply(lambda x: f'{x:.2f}')
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()
        return df


    # compare between 2 different Stores based on dataframe index (start/stop)

    def compare_Store(df1, df2, i):

        data1 = df1.iloc[i, :]
        df1_dict = data1.to_dict()

        data2 = df2.iloc[0, :]
        df2_dict = data2.to_dict()

        holiDay = {0: 'No', 1: 'Yes'}
        Type = {1: 'A', 2: 'B', 3: 'C'}
        st.code(f'''{df1_dict['Store']} : Store : {df2_dict['Store']}           {Type[df1_dict['Type']]} : Type : {Type[df2_dict['Type']]}           {df1_dict['Size']} : Size : {df2_dict['Size']}           {holiDay[df1_dict['IsHoliday']]}  :  HoliDay : {holiDay[df2_dict['IsHoliday']]}''')

        comparison.vertical_line()
        col1, col2 = st.columns([0.55, 0.45])

        with col1:

            for i in ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                p, c = float(df1_dict[i]), float(df2_dict[i])

                if p != 0:
                    diff = ((c-p)/p)*100
                    col3, col4, col5, col6 = st.columns([0.3, 0.4, 0.25, 0.05])
                    with col3:
                        add_vertical_space(1)
                        st.markdown(f"#### {float(df1_dict[i]):.2f}")

                    with col4:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                                    No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col5:
                        add_vertical_space(1)
                        st.markdown(f"#### {float(df2_dict[i]):.2f}")

                else:
                    col3, col4, col5, col6 = st.columns([0.3, 0.4, 0.25, 0.05])
                    with col3:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col4:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                                    No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{c*100:.2f}%")

                    with col5:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")

        with col2:

            for i in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
                p, c = df1_dict[i], df2_dict[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col7, col8, col9, col10 = st.columns(
                        [0.05, 0.3, 0.4, 0.25])
                    with col8:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col9:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                                    No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col10:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")

        add_vertical_space(3)


    # compare between selected Store and Top 10 Stores

    def compare_with_top_Stores(df1, df2):

        Store_list = df1['Store'].tolist()

        user_Store = df2['Store'].tolist()[0]

        if user_Store in Store_list:

            if Store_list[0] == user_Store:
                col1, col2, col3 = st.columns([0.29, 0.42, 0.29])
                with col2:
                    st.info('The Selected Store Ranks Highest in Weekly Sales')
            
            else:
                user_Store_index = Store_list.index(user_Store)
                for i in range(0, user_Store_index):
                    comparison.compare_Store(df1,df2,i)

        else:

            for i in range(1, 10):
                comparison.compare_Store(df1,df2,i)


    # compare between selected week and bottom 10 Stores - filter options

    def bottom_Store_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            Year = st.selectbox(label='Year  ', options=comparison.Year())

        with col2:
            Month = st.selectbox(
                label='Month  ', options=comparison.Month(Year))

        with col1:
            Day = st.selectbox(
                label='Day  ', options=comparison.Day(Month, Year))

        with col4:
            Store = st.selectbox(
                label='Store  ', options=comparison.Store(Day, Month, Year))

        return Day, Month, Year, Store


    # Store wise Weekly_Sales and remaining columns avg from sales table

    def bottom_Store_sales(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store, Type, sum(Weekly_Sales) as Weekly_Sales,  
                        Size, IsHoliday, avg(Temperature) as Temperature,  
                        avg(Fuel_Price) as Fuel_Price, avg(MarkDown1) as MarkDown1,  
                        avg(MarkDown2) as MarkDown2, avg(MarkDown3) as MarkDown3,  
                        avg(MarkDown4) as MarkDown4, avg(MarkDown5) as MarkDown5, 
                        avg(CPI) as CPI, avg(Unemployment) as Unemployment
                       
                        from sales
                        where {condition}
                        group by Store, Type, Size, IsHoliday               
                        order by Weekly_Sales desc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]
        df = pd.DataFrame(s, columns=columns, index=index)
        df['Weekly_Sales'] = df['Weekly_Sales'].apply(lambda x: f'{x:.2f}')
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()
        return df


    # compare between selected Store and Bottom 10 Stores

    def compare_with_bottom_Stores(df1, df2):

        Store_list = df1['Store'].tolist()

        user_Store = df2['Store'].tolist()[0]

        if user_Store in Store_list:

            if Store_list[-1] == user_Store:
                col1, col2, col3 = st.columns([0.30, 0.40, 0.30])
                with col2:
                    st.info('The Selected Store Ranks Lowest in Weekly Sales')
            
            else:
                user_Store_index = Store_list.index(user_Store)
                for i in range(user_Store_index+1, 10):
                    comparison.compare_Store(df1,df2,i)

        else:

            for i in range(1, 10):
                comparison.compare_Store(df1,df2,i)



class features:

    # Store wise Weekly_Sales and remaining columns avg from sales table
 
    def sql_sum_avg(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store, Type, sum(Weekly_Sales) as Weekly_Sales,  
                        Size, IsHoliday, avg(Temperature) as Temperature,  
                        avg(Fuel_Price) as Fuel_Price, avg(MarkDown1) as MarkDown1,  
                        avg(MarkDown2) as MarkDown2, avg(MarkDown3) as MarkDown3,  
                        avg(MarkDown4) as MarkDown4, avg(MarkDown5) as MarkDown5, 
                        avg(CPI) as CPI, avg(Unemployment) as Unemployment
                        
                        from sales
                        where {condition}
                        group by Store, Type, Size, IsHoliday               
                        order by Weekly_Sales desc''')

        s = cursor.fetchall()
        
        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()

        return df


    # Store list from sales table

    def Store():
        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store from sales''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Year list from sales table

    def Year():

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Year from sales
                           order by Year asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Month list from sales table based on Year

    def Month(Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Month from sales
                           where Year='{Year}'
                           order by Month asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # Day list from sales table based on Year,Month

    def Day(Month, Year):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Day from sales
                           where Year='{Year}' and Month='{Month}'
                           order by Day asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # features tab filter options

    def filter_options():

            col1, col2, col3 = st.columns(3, gap='medium')

            with col3:
                Year = st.selectbox(label='Year ', options=features.Year())

            with col2:
                Month = st.selectbox(label='Month ', options=features.Month(Year))

            with col1:
                Day = st.selectbox(label='Day ', options=features.Day(Month, Year))

            return Day, Month, Year


    # sales table based on condition

    def sql(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select * from sales
                           where {condition}''')

        s = cursor.fetchall()
            
        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        mydb.close()
        return df


    # create 10 bins based on range values [like 20-40, 40-60, etc.,]

    def bins(df, feature):
        
        # filter 2 columns ---> like Temperature and Weekly_Sales
        df1 = df[['Weekly_Sales',feature]]

        # Calculate bin edges
        bin_edges = pd.cut(df1[feature], bins=10, labels=False, retbins=True)[1]

        # Create labels for the bins
        bin_labels = [f'{f"{bin_edges[i]:.2f}"} to <br>{f"{bin_edges[i+1]:.2f}"}' for i in range(0, len(bin_edges)-1)]

        # Create a new column by splitting into 10 bins
        df1['part'] = pd.cut(df1[feature], bins=bin_edges, labels=bin_labels, include_lowest=True)

        return df1


    # holiDay (yes and no) and avg weekly sales from sales table

    def sql_holiDay(condition):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select IsHoliday, avg(Weekly_Sales) as Weekly_Sales
                           from sales
                           where {condition}
                           group by IsHoliday''')

        s = cursor.fetchall()
            
        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')
        df['Weekly_Sales'] = df['Weekly_Sales'].apply(lambda x: f"{x:.2f}")
        df['decode'] = df['IsHoliday'].apply(lambda x: 'Yes' if x==1 else 'No')
        df.drop(columns=['IsHoliday'], inplace=True)

        cursor.close()
        mydb.close()
        return df


    # weekly sales bar chart based on 10 bins

    def Store_features(df):

        columns = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                   'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
        
        color = ['#5D9A96','#5cb85c','#5D9A96','#5cb85c',
                 '#5D9A96','#5cb85c','#5D9A96','#5cb85c','#5D9A96']
        
        
        c = 0
        for i in columns:

            # create 10 bins based on range values [like 20-40, 40-60, etc.,]
            df1 = features.bins(df=df, feature=i)

            # group unique values and sum Weekly_Sales
            df2 = df1.groupby('part')['Weekly_Sales'].sum().reset_index()

            # only select weekly sales greater than zero (less than zero bins automatically removed and it can't show barchart)
            df2 = df2[df2['Weekly_Sales']>0]

            # barchart with df2 dataframe values
            plotly.vertical_bar_chart(df=df2, x='part', y='Weekly_Sales',
                                      color=color[c], text='part',
                                      title_x=0.40, title=columns_dict()[i])
            
            c += 1
            add_vertical_space(2)



class prediction:

    # Type and Size dictionary based on Store from sales table --> {Store:Type},{Store:Size}

    def Type_Size_dict():

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Store, Type, Size 
                           from sales
                           group by Store, Type, Size
                           order by Store asc;''')

        s = cursor.fetchall()

        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns)

        Store = df['Store'].to_list()
        Type = df['Type'].to_list()
        Size = df['Size'].to_list()

        Type_dict, Size_dict = {}, {}

        for i in range(0, len(Store)):
            Type_dict[Store[i]] = Type[i]
            Size_dict[Store[i]] = Size[i]

        cursor.close()
        mydb.close()

        return Type_dict, Size_dict


    # department list from sales table based on Store

    def Dept(Store):

        mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='vino8799',
                                database='retail_forecast')
        cursor = mydb.cursor()

        cursor.execute(f'''select distinct Dept from sales
                           where Store='{Store}'
                           order by Dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        mydb.close()

        return data


    # get input data from users and predict Weekly_Sales
    
    def predict_Weekly_Sales():

        # get input from users
        with st.form('prediction'):

            col1, col2, col3 = st.columns([0.5, 0.1, 0.5])

            with col1:

                user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                          max_value=date(2013, 12, 31), value=date(2010, 2, 5))

                Store = st.number_input(label='Store', min_value=1, max_value=45,
                                        value=1, step=1)

                Dept = st.selectbox(label='Department',
                                    options=prediction.Dept(Store))

                holiDay = st.selectbox(label='HoliDay', options=['Yes', 'No'])

                Temperature = st.number_input(label='Temperature(°F)', min_value=-10.0,
                                              max_value=110.0, value=-7.29)

                Fuel_Price = st.number_input(label='Fuel Price', max_value=10.0,
                                             value=2.47)

                CPI = st.number_input(label='CPI', min_value=100.0,
                                      max_value=250.0, value=126.06)

            with col3:

                MarkDown1 = st.number_input(label='MarkDown1', value=-2781.45)

                MarkDown2 = st.number_input(label='MarkDown2', value=-265.76)

                MarkDown3 = st.number_input(label='MarkDown3', value=-179.26)

                MarkDown4 = st.number_input(label='MarkDown4', value=0.22)

                MarkDown5 = st.number_input(label='MarkDown5', value=-185.87)

                Unemployment = st.number_input(label='Unemployment',
                                               max_value=20.0, value=3.68)

                add_vertical_space(2)
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()

        # user entered the all input values and click the button
        if button:
            with st.spinner(text='Processing...'):

                # load the regression pickle model
                with open(r'V:\project\vk_project\retail_sales\model1_markdown.pkl', 'rb') as f:
                    model = pickle.load(f)

                holiDay_dict = {'Yes': 1, 'No': 0}
                Type_dict, Size_dict = prediction.Type_Size_dict()

                # make array for all user input values in required order for model prediction
                user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                       Store, Dept, Type_dict[Store], Size_dict[Store],
                                       holiDay_dict[holiDay], Temperature,
                                       Fuel_Price, MarkDown1, MarkDown2, MarkDown3,
                                       MarkDown4, MarkDown5, CPI, Unemployment]])

                # model predict the selling price based on user input
                y_pred = model.predict(user_data)[0]

                # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
                Weekly_Sales = f"{y_pred:.2f}"

                return Weekly_Sales 
        




streamlit_config()


with st.sidebar:

    add_vertical_space(1)
    option = option_menu(menu_title='', options=['Home','Migrating to SQL', 'Top Sales', 'Comparison', 'Features', 'Prediction', 'Exit'],
                         icons=['database-fill', 'bar-chart-line', 'globe', 'list-task', 'slash-square', 'sign-turn-right-fill'])
    
    col1, col2, col3 = st.columns([0.26, 0.48, 0.26])
    with col2:
        button = st.button(label='Submit')



if button and option == 'Migrating to SQL':

    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

    with col2:

        add_vertical_space(2)

        with st.spinner('Dropping the Existing Table...'):
            sql.drop_table()
        
        with st.spinner('Creating Sales Table...'):
            sql.create_table()
        
        with st.spinner('Migrating Data to SQL Database...'):
            sql.data_migration()

        st.success('Successfully Data Migrated to SQL Database')
        st.balloons()



elif option == 'Top Sales':

    tab1, tab2 = st.tabs(['Top Stores', 'Top Departments'])

    with tab1:

        Day1, Month1, Year1, Dept1 = top_sales.top_Store_filter_options()
        add_vertical_space(3)

        if Dept1 == 'Overall':
            df1 = top_sales.top_Store_sales(
                f"Day='{Day1}' and Month='{Month1}' and Year='{Year1}'")

            plotly.vertical_bar_chart(df=df1, x='Store_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#00BFFF', title='Top Stores in Weekly Sales', title_x=0.35)

        else:
            df1 = top_sales.top_Store_sales(
                f"Day='{Day1}' and Month='{Month1}' and Year='{Year1}' and Dept='{Dept1}'")

            plotly.vertical_bar_chart(df=df1, x='Store_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#00BFFF', title='Top Stores in Weekly Sales', title_x=0.35)


    with tab2:

        Day2, Month2, Year2, Store2 = top_sales.top_Dept_filter_options()
        add_vertical_space(3)

        if Store2 == 'Overall':
            df2 = top_sales.top_Dept_sales(
                f"Day='{Day2}' and Month='{Month2}' and Year='{Year2}'")

            plotly.vertical_bar_chart(df=df2, x='Dept_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#00BFFF', title='Top Departments in Weekly Sales', title_x=0.35)

        else:
            df2 = top_sales.top_Dept_sales(
                f"Day='{Day2}' and Month='{Month2}' and Year='{Year2}' and Store='{Store2}'")

            plotly.vertical_bar_chart(df=df2, x='Dept_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#00BFFF', title='Top Departments in Weekly Sales', title_x=0.35)



elif option == 'Comparison':

    tab1, tab2, tab3, tab4 = st.tabs(['Previous Week', 'Top Stores', 
                                      'Bottom Stores','Manual Comparison'])

    with tab1:

        Day, Month, Year, Store, Dept = comparison.previous_week_filter_options()
        add_vertical_space(3)

        df = comparison.sql(f"Store='{Store}' and Dept='{Dept}'")

        comparison.previous_week_sales_comparison(df, Day, Month, Year)


    with tab2:

        # user input filter options
        Day3, Month3, Year3, Store3 = comparison.top_Store_filter_options()
        add_vertical_space(3)

        # sql query filter the data based on user input Day,Month,Year, Store
        df3 = comparison.top_Store_sales(f"""Day='{Day3}' and Month='{Month3}' and 
                                 Year='{Year3}' and Store='{Store3}'""")

        # sql query calculte the sum of weekly sales in desc order (1 to 45) all Stores
        df4 = comparison.top_Store_sales(f"""Day='{Day3}' and Month='{Month3}' and 
                                 Year='{Year3}'""")

        # top 10 Stores in weekly sales
        df_top10 = df4.iloc[:10, :]

        # user selected Store compare to top 10 Stores based on weekly sales
        comparison.compare_with_top_Stores(df_top10, df3)


    with tab3:

        # user input filter options
        Day4, Month4, Year4, Store4 = comparison.bottom_Store_filter_options()
        add_vertical_space(3)

        # sql query filter the data based on user input Day,Month,Year, Store
        df5 = comparison.bottom_Store_sales(f"""Day='{Day4}' and Month='{Month4}' and 
                                 Year='{Year4}' and Store='{Store4}'""")

        # sql query calculte the sum of weekly sales in desc order (1 to 45) all Stores
        df6 = comparison.bottom_Store_sales(f"""Day='{Day4}' and Month='{Month4}' and 
                                 Year='{Year4}'""")

        # bottom 10 Stores in weekly sales
        df_bottom10 = df6.iloc[-10:, :]

        # user selected Store compare to top 10 Stores based on weekly sales
        comparison.compare_with_bottom_Stores(df_bottom10, df5)


    with tab4:

        Day1, Month1, Year1, Store1, Dept1, Day2, Month2, Year2, Store2, Dept2 = comparison.manual_filter_options()
        add_vertical_space(3)

        df1 = comparison.sql(f"""Day='{Day1}' and Month='{Month1}' and Year='{Year1}' and 
                                 Store='{Store1}' and Dept='{Dept1}'""")

        df2 = comparison.sql(f"""Day='{Day2}' and Month='{Month2}' and Year='{Year2}' and 
                                 Store='{Store2}' and Dept='{Dept2}'""")

        comparison.manual_comparison(df1, df2)



elif option == 'Features':

    tab1,tab2 = st.tabs(['Date', 'Store'])
    
    with tab1:

        Day,Month,Year = features.filter_options()

        # sum of weekly sales and avg of remaining values from sales table
        df = features.sql_sum_avg(f"""Day='{Day}' and Month='{Month}' and Year='{Year}'""")
        add_vertical_space(2)


        columns = ['Size', 'Type', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
                'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
        
        for i in columns:
            plotly.scatter_chart(df=df,x=i,y='Weekly_Sales')


    with tab2:

        col1,col2,col3 = st.columns(3)

        with col1:
            Store = st.selectbox(label='Store', options=features.Store())
        
        add_vertical_space(2)

        df = features.sql(f'Store={Store}')

        holiDay_df = features.sql_holiDay(f'Store={Store}')

        plotly.pie_chart(df=holiDay_df, x='decode', y='Weekly_Sales',
                         title='HoliDay', title_x=0.40)

        features.Store_features(df)



elif option == 'Prediction':

    Weekly_Sales = prediction.predict_Weekly_Sales()

    if Weekly_Sales:

        # apply custom css style for prediction text
        style_prediction()

        st.markdown(f'### <div class="center-text">Predicted Sales = {Weekly_Sales}</div>', 
                    unsafe_allow_html=True)

        st.balloons()



elif option == 'Exit':
    
    add_vertical_space(2)

    col1,col2,col3 = st.columns([0.20,0.60,0.20])

    with col2:

        st.success('#### Thank you for your time. Exiting the application')
        st.balloons()

elif option ==   "Home":

    def load_lottieurl(url):

        r = requests.get(url)

        if r.status_code != 200:

            return None
        
        return r.json()

    # Use local CSS
    def local_css(file_name):

        with open(file_name) as f:

            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css(r"V:\project\vk_project\industrial_copper_modeling\style.css")

    lottie_coding = lottie(r"V:\project\vk_project\lottiite animation\intro vk.json")


    # ---- HEADER SECTION -----``
    with st.container():

        col1,col2=st.columns(2)

        with col1:

            st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> Hi,  </span><span style='color: white;'> I am vinoth kumar </h1>",unsafe_allow_html=True)
            
            st.markdown(
                f"<h1 style='font-size: 40px;'><span style='color: white;'>A Data Scientist,</span><span style='color: #00BFFF;'> From India</span></h1>",
                unsafe_allow_html=True
                )
            
            st.write(f'<h1 style="color:#B0C4DE; font-size: 20px;">A data scientist skilled in extracting actionable insights from complex datasets, adept at employing advanced analytics and machine learning techniques to solve real-world problems. Proficient in Python, statistical modeling, and data visualization, with a strong commitment to driving data-driven decision-making.</h1>', unsafe_allow_html=True)    

            st.write("[view more projects >](https://github.com/vinothkumarpy?tab=repositories)")

        with col2:

            st_lottie(lottie_coding, height=400, key="coding")    

    # ---- WHAT I DO ----
    with st.container():

        st.write("---")

        col1,col2,col3=st.columns(3)

        with col1:

            file = lottie(r"V:\project\vk_project\lottiite animation\speak with ropot.json")
            st_lottie(file,height=500,key=None)

        with col2:

            st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> WHAT  </span><span style='color: white;'> I DO </h1>",unsafe_allow_html=True)
        
        with col3:

            file = lottie(r"V:\project\vk_project\lottiite animation\ml process.json")
            st_lottie(file,height=500,key=None)    

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'>Data  </span><span style='color: white;'>Preprocessing:</h1>",unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Data Understanding : The dataset includes information on store attributes, sales, and various features, such as store name, department, date, type, size, weekly sales, and environmental factors like holiday status, temperature, fuel price, markdowns, CPI, and unemployment. The main goal is to predict weekly sales, which is the target variable for our models. This initial exploration lays the groundwork for data preprocessing and model development.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Encoding and Data Type Conversion : The process involves converting categorical features into numerical representations and aligning data types with modeling requirements. This ensures effective use of categorical information in later project stages.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Handling Null Values : The "MarkDown" columns have over 50% null values, unlike other columns which have few nulls. To fix this, we use machine learning models to predict and fill in the missing values, creating a more complete and reliable dataset for further analysis and modeling. This approach helps improve the overall quality of our dataset.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Feature Improvement : To improve our model, we focus on refining the dataset by creating new features for deeper insights and better efficiency. Using Seaborn Heatmap, we found that only Size and Type have some correlation with weekly sales (0.21 and 0.17, respectively). This highlights the need to enhance features strategically to boost our models predictive power.</h1>', unsafe_allow_html=True)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Machine Learning </span><span style='color: white;'>Regression Model</h1>",unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Multiple Models : Due to over 50% null values in the MarkDown columns, we train two machine learning models to predict weekly sales: one using MarkDown features and one without. This approach helps us understand the impact of MarkDown on prediction accuracy.</h1>', unsafe_allow_html=True) 
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Assessment : Our goal is to predict weekly sales using regression. We split the dataset into training and testing subsets, applying various algorithms and evaluating them with the R2 (R-squared) metric to identify the best one.</h1>', unsafe_allow_html=True) 
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Selection : The Extra Trees Regressor and Random Forest Regressor show strong performance without overfitting. I choose the Random Forest Regressor for its balance between interpretability and accuracy.</h1>', unsafe_allow_html=True) 
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Accuracy and Metrics : After optimizing parameters, model1 (with MarkDowns) and model2 achieve accuracies of 97.4% and 97.7%, respectively. Model1 is chosen for its robust predictions on unseen data. Key metrics like mean absolute error, mean squared error, root mean squared error, and R-squared further validate the models reliability.</h1>', unsafe_allow_html=True) 
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Persistence : The final model is saved to a pickle file, allowing easy future use for predicting weekly sales.</h1>', unsafe_allow_html=True) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Exploratory Data Analysis (EDA) </span><span style='color: white;'>-Streamlit Application</h1>",unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Migrating to SQL : The weekly sales predictions data is stored in a PostgreSQL database, enabling efficient data retrieval and analysis using SQL queries.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Top Sales : Top Stores: View the top 10 stores by weekly sales, selectable by date and overall or department-specific view. Top Departments: View the top 10 departments by weekly sales, customizable by date and overall or store-specific perspective.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Comparison : Analyze weekly sales dynamics by comparing key features such as type, size, holiday, temperature, fuel price, CPI, unemployment, and markdowns. This includes comparisons of current and previous weeks, top and bottom-performing stores, and manual evaluations of different stores and departments.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Features : Understand how different feature ranges affect weekly sales, considering both date-wise and store-wise perspectives. Features include type, size, holiday, temperature, fuel price, CPI, unemployment, and markdowns.</h1>', unsafe_allow_html=True)
        
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Prediction : Input parameters such as date, store, department, holiday, temperature, fuel price, CPI, unemployment, and markdowns to predict weekly sales using a pre-trained Random Forest Regressor model. This allows users to experiment with different inputs for personalized sales forecasts.</h1>', unsafe_allow_html=True)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Interactive  </span><span style='color: white;'>Streamlit UI</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">Crafted an engaging and user-friendly interface for seamless data exploration and presentation.</h1>', unsafe_allow_html=True)

        st.markdown("[🔗 GitHub Repo >](https://github.com/vinothkumarpy/Retail-sales.git)")    



    with st.container():

        st.write("---")

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Used-Tech  </span><span style='color: white;'>& Skills</h1>",unsafe_allow_html=True)

        col1,col2,col3 =st.columns(3)

        with col1:

            file = lottie(r"V:\project\vk_project\lottiite animation\python.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>python</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

            file = lottie(r"V:\project\vk_project\lottiite animation\pandas.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Pandas</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)
            

            file = lottie(r"V:\project\vk_project\lottiite animation\data_exploaration.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Data Exploaration</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)
            
            file = lottie(r"V:\project\vk_project\retail_sales\data visu.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Data Visualization</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

        with col2:

            file = lottie(r"V:\project\vk_project\lottiite animation\numpy.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Numpy</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)


            file = lottie(r"V:\project\vk_project\retail_sales\mysql.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>SQL</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

            file = lottie(r"V:\project\vk_project\lottiite animation\working with data set.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Scikit-Learn</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)
            

        with col3: 
            
            file = lottie(r"V:\project\vk_project\lottiite animation\ml train.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Mechine-Learning Model (Regression)</h1>", unsafe_allow_html=True)
            st_lottie(file,height=500,key=None)


            file = lottie(r"V:\project\vk_project\lottiite animation\frame work.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Web application development with Streamlit</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

    # ---- PROJECTS ----
    with st.container():

        st.write("---")
        st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> About  </span><span style='color: white;'> Projects </h1>",unsafe_allow_html=True)
        
        col1,col2=st.columns(2)

        with col1:

            file = lottie(r"V:\project\vk_project\retail_sales\retail.json")
            st_lottie(file,height=400,key=None)

        with col2:

            st.write("##")
            st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">The project Enhance Retail Sales Forecast employs advanced machine learning techniques, prioritizing careful data preprocessing, feature enhancement, and comprehensive algorithm assessment and selection. The streamlined Streamlit application integrates Exploratory Data Analysis (EDA) to find trends, patterns, and data insights. It offers users interactive tools to explore top-performing stores and departments, conduct insightful feature comparisons, and obtain personalized sales forecasts. With a commitment to delivering actionable insights, the project aims to optimize decision-making processes within the dynamic retail landscape</h1>', unsafe_allow_html=True)
        
    # ---- CONTACT ----
    with st.container():

        st.write("---")
        st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> Get In Touch  </span><span style='color: white;'> With Me </h1>",unsafe_allow_html=True)
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/vinoharish8799@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit" style="background-color: #00BFFF; color: white;">Send</button>
        </form>
        """

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column: 
            st.empty()   
