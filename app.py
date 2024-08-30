import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(layout="wide", page_title="Real Estate Report", page_icon="🏠")

st.header("Real Estate Report per US Market 🏠", divider= 'blue')

st.sidebar.title("Main Report Filters")

######
cashflow = pd.read_csv('data/cashflows.csv', index_col=0, parse_dates=True).drop(columns=['date_cashflow'])
cashflow['state'] = cashflow['market'].str.split(',').str[1]
cashflow['city'] = cashflow['market'].str.split(',').str[0]

equilibrium = pd.read_csv('data/equilibriums.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])
equilibrium['state'] = equilibrium['market'].str.split(',').str[1]
equilibrium['city'] = equilibrium['market'].str.split(',').str[0]

general = pd.read_csv('data/general_parameters.csv', index_col=0)
summary = pd.read_csv('data/summaries.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])

table = pd.read_csv('data/tables.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])
table['state'] = table['market'].str.split(',').str[1]
table['city'] = table['market'].str.split(',').str[0]

##3 Market Filters
states = summary['state'].unique()
classes = summary['slice'].unique()

filtro_columnas_mapa = ['irr','equity_multiple','current_price','loan','equity','market_cagr',
                        'noi_cap_rate_compounded','operation_cashflow','market_cap_appreciation_bp','npv','npv/equity','demand_vs_supply','demand_yoy_growth','supply_yoy_growth']
mapa_columns = pd.DataFrame(filtro_columnas_mapa, columns=['columnas'])
mapa_columns.index = mapa_columns['columnas'].str.replace('_',' ').replace('bp','basis point').str.title().str.replace('Yoy','YoY%').str.replace('Npv','Net Present Value').str.replace('Irr', 'IRR')
mapa_columns['unit'] = ['%','x','USD','USD','USD','%','%','%','bp','USD','x','%','%','%']






def filter_summary(states_f, city_f, slice_f):
    return summary[(summary['state'] == states_f) & (summary['city'] == city_f) & (summary['slice'] == slice_f)].copy()

with st.expander('States Filter'):

    st.session_state.selected_states = states
    selected_states = st.multiselect('**Select States**', states, default=st.session_state.selected_states)


##3 default all selected
with st.sidebar:

    ################################################################################

    slice = st.selectbox('Select Class', options=classes, index=0)
    st.session_state.slice = slice
    box_selector_pop = ['All','+100K', '+500K', '+1M', '+2M' , '+3M' , '+5M', '+7M', '+10M']
    population = st.selectbox('Select Population', options=box_selector_pop, index=2)
    result = population.replace('+', '').replace('.', '').replace('M', '000000').replace('K', '000').replace('All', '0')
    st.session_state.population = result

    filtro_columnas_mapa_mostrar = st.selectbox('Aspect to classify', options=mapa_columns.index, index=0)
    st.session_state.filtro_columnas_mapa = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[0]

    
    if st.button('Reset Filters'):
        st.experimental_rerun()

    st.markdown('--'*20)

    ################################################################################
    st.subheader(' Indivual Market Filters')
    selected_state = st.selectbox('Select State', selected_states, index=0)
    st.session_state.selected_state = selected_state

    selected_cities = summary[summary['state'] == selected_state]['city'].unique()
    selected_city = st.selectbox('Select City', selected_cities, index=0)
    st.session_state.selected_city = selected_city

    selected_slices_city = summary[(summary['state'] == selected_state) & (summary['city'] == selected_city)]['slice'].unique()
    selected_slice_city = st.selectbox('Select Class', selected_slices_city, index=0, key='slice_city')
    st.session_state.selected_slice_city = selected_slice_city


    if 'saved_searches' not in st.session_state:
        st.session_state.saved_searches = []

    if st.button('Save Search'):
        search = {
            'state': selected_state,
            'city': selected_city,
            'slice': selected_slice_city
        }
        st.session_state.saved_searches.append(search)
        st.success(f"Search saved: {selected_state}, {selected_city}, {selected_slice_city}")

    if st.button('Clear Saved Searches'):
        st.session_state.saved_searches = []
        st.success('All saved searches have been cleared.')



## ELIJO LA COLUMNA POR LA CUAL QUIERO FILTRAR
summary_filtered = summary[(summary['slice'] == st.session_state.slice) & (summary['state'].isin(selected_states)) & (summary['population'] >= int(st.session_state.population))].copy()
summary_filtered['npv/equity'] = summary_filtered['npv'] / summary_filtered['equity']
if summary_filtered.empty:
    st.error('No data available for the selected filters')
    st.stop()


unidad_columna = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[1]

def transformar_value(column , unidad): 
    if unidad == '%':
        return (column).round(4) 
    elif unidad == 'x':
        return column.round(2)
    elif unidad == 'USD':
        return column.round(0)
    elif unidad == 'bp':
        return column
    
def valor_a_mostrar(column, unidad):
    if unidad == '%':
        return f'{column:.2%}'
    elif unidad == 'x':
        return f'{column:.2f}x'
    elif unidad == 'USD':
        return f'USD {column:,.0f}'
    elif unidad == 'bp':
        return f'{column:.0f} bp'
    
# Función para asignar colores basados en IRR
def get_color(value, column_name):
    #if value >  column_name.quantile(0.85):
    if value > 90:
        return [0, 128, 0, 200]  # Verde más oscuro
    #elif value > column_name.quantile(0.7):
    elif value > 75:
        return [144, 238, 144, 200]  # Verde super claro
    #elif value > column_name.quantile(0.4):
    elif value > 50:
        # verde limon
        return [173, 200, 47, 200]
    #elif value > column_name.quantile(0.1):
    elif value > 30:
        # amarillo
        return [255, 255, 0, 200]
    elif value > 5:
        # naranja
        return [255, 165, 0, 200] 
    #elif value <= column_name.quantile(0.1):
    elif value <= 5:
        # rojo
        return [255, 0, 0, 200]
# Añadimos una nueva columna para los colores

summary_filtered['value'] = transformar_value(summary_filtered[st.session_state.filtro_columnas_mapa], unidad_columna)
summary_filtered['value_show'] = summary_filtered['value'].apply(lambda x: valor_a_mostrar(x, unidad_columna))
##3 para el alto de la barra hago un rank y cada uno le asigno su puesto siendo 100 el mayor valor y 1 el menor
summary_filtered['bar_height'] =  summary_filtered['value'].rank(ascending=True, method='max', pct=True) * 100
summary_filtered['color'] = summary_filtered['bar_height'].apply( lambda x: get_color(x, summary_filtered['bar_height']))
summary_filtered['color_no_list'] = summary_filtered['color'].apply(lambda x: f'rgba({x[0]},{x[1]},{x[2]},{x[3]})')
#summary_filtered['bar_height'] = summary_filtered['bar_height'] ** 1.8
summary_filtered['market'] = summary_filtered['market'].str.replace(',', ' - ')
summary_filtered['log_population'] = (summary_filtered['population']) ** 0.5
summary_filtered['population_millions'] = (summary_filtered['population'] / 1_000_000).round(2).astype(str) + 'M'


###########
col1 , col2  = st.columns([1.8, 1.1])

with col1:
    #st.subheader(' per US Market + demographic aggregation for ' + st.session_state.slice + ' buildings')
    st.subheader( f'US Markets classified by {filtro_columnas_mapa_mostrar} + population {st.session_state.slice} buildings')
    # Definimos la capa ColumnLayer
    ### reset view for map 
    st.button('Reset View')
    irr_layer = pdk.Layer(
        "ColumnLayer",
        data=summary_filtered,
        get_position=["longitude", "latitude"],
        get_elevation="bar_height",
        elevation_scale=2500,  # Ajusta según sea necesario para la visibilidad
        radius=20000,  # Ajusta el radio de las columnas
        get_fill_color="color",  # Asignar color basado en la columna calculada
        pickable=True,
        extruded=True,
        auto_highlight=True
    )

    population_layer = pdk.Layer(
        "ScatterplotLayer",
        data=summary_filtered,
        get_position=["longitude", "latitude"],
        get_radius="log_population",  # Radio proporcional a la población
        radius_scale=90,  # Ajustar el factor de escala según sea necesario
        get_fill_color= ## Verde con transparencia
        [55, 8, 94, 60],
        pickable=True
    )

    # Configura la vista inicial del mapa
    view_state = pdk.ViewState(
        longitude=-99,
        latitude=38.83,
        zoom=3.4,
        min_zoom=2,
        max_zoom=7,
        pitch=75,  # Reducido para hacer más distinguibles las barras altas
        bearing=23
    )

    lights = pdk.LightSettings(
        number_of_lights= 3)


    # Renderiza el mapa
    st.pydeck_chart(
        pdk.Deck(
            #map_style="mapbox://styles/mapbox/light-v9",
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=view_state,
            layers=[irr_layer, population_layer],
            tooltip={
                #"html": "<b>City:</b> {city}<br/><b>IRR:</b> {IRR_percentage}%<br/><b>Population:</b> {population}",
                "html": """
                    <b>State:</b> {state}<br/>
                    <b>City:</b> {city}<br/>
                    <b>Value:</b> {value_show}<br/>
                    <b>Population:</b> {population_millions}
                    """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "14px",
                    "padding": "10px",
                    "borderRadius": "15px"
                }
            }
            ,

            # width= 200,
            # height=500
        ),
        use_container_width=True
        )




with col2:
    subcol1, subcol2 = st.columns([0.2, 1])
    with subcol2:
        st.subheader('Distribution')
        fig = px.histogram(
            summary_filtered, 
            x='value', 
            color='color_no_list', 
            color_discrete_map='identity',
            pattern_shape='city',
            orientation='v', 
            nbins=50,  
            ###3 avoid showing number of observations in the hover
            template='plotly_dark',
        )

        # Ajustar la apariencia del gráfico
        fig.update_layout(
            showlegend=False, 
            yaxis_visible=False, 
            xaxis_title=f'{filtro_columnas_mapa_mostrar}', 
            xaxis_tickformat=',.2f' if unidad_columna == 'USD' else '.2%' if unidad_columna == '%' else '.2f',
            yaxis_title=None,
            bargap=0.1  # Controlar el espacio entre las barras del histograma
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # Notas del mapa
        st.markdown('''**Map Notes:** The height of the bars is proportional to the value you're filtering, and the color is based on percentiles.  
                The size of the circles is proportional to the population of the city.''')


st.subheader('US Markets Summary', divider= 'blue')

data_show = summary_filtered[['market', 'current_price','market_cagr','noi_cap_rate_compounded',
                               'fixed_interest_rate', 'operation_cashflow','market_cap_appreciation_bp', 'irr', 
                                'npv', 'npv/equity',  'equity_multiple', 'demand_vs_supply',
                                'demand_yoy_growth', 'supply_yoy_growth']]
data_show = data_show.sort_values( st.session_state.filtro_columnas_mapa, ascending=False)
data_show.columns = ['Market', 'Current Price','Market CAGR', 'NOI Cap Rate Compounded',
                        'Fixed Interest Rate', 'Operation Cashflow', 'M. Cap BP', 'IRR', 'NPV', 'NPV/Equity', 'Equity Multiple', 'Demand vs Supply', 'Demand YoY Growth', 'Supply YoY Growth']

for col in ['Current Price','NPV']:
    data_show[col] = data_show[col].apply(lambda x: f'${x:,.0f}')

for col in ['Market CAGR', 'NOI Cap Rate Compounded', 'Fixed Interest Rate', 'Operation Cashflow', 'IRR',]:
    data_show[col] = data_show[col].apply(lambda x: f'{x:.2%}')

data_show['Demand vs Supply'] = data_show['Demand vs Supply'].apply(lambda x: f'{x:.2%}')
data_show['Demand YoY Growth'] = data_show['Demand YoY Growth'].apply(lambda x: f'{x:.2%}')
data_show['Supply YoY Growth'] = data_show['Supply YoY Growth'].apply(lambda x: f'{x:.2%}')


### header in blue and background in white
st.dataframe(data_show.set_index('Market'), use_container_width=True)

###3 filtros por estado y mercado singulares para examinar por separado
st.header('Individual Market Analysis 🏠', divider= 'blue')

with st.expander('Saved Searches'):
    st.subheader(' Summary Comparison', divider= 'blue')
    st.write('Summary for selected markets')

    comparison = pd.DataFrame()
    for search in st.session_state.saved_searches:
        comparison = pd.concat([comparison, filter_summary(search['state'], search['city'], search['slice'])], axis=0)

    if comparison.empty:
        st.error('No data available, save some filters to compare')
    else:
        comparison['LTV'] = '60%'
        comparison['5yrs Swap Rate'] = '3.45%'
        comparison['Spread'] = '3.0%'
        comparison['Net Int Rate'] = '6.45%'
        comparison['Years'] = '5'

        # Selección de columnas sin duplicación
        comparison_ = comparison[['slice', 'market', 'current_price', 'LTV', 'loan', 'equity',
                                '5yrs Swap Rate', 'Spread', 'Net Int Rate', 'Years', 'market_cagr',
                                'noi_cap_rate_compounded', 'operation_cashflow', 
                                'market_cap_appreciation_bp', 'irr', 'equity_multiple']].set_index(['slice', 'market'])

        # Renombrar las columnas de manera correcta
        comparison_.columns = ['Current Price', 'LTV', 'Loan', 'Equity', '5yrs Swap Rate', 'Spread', 
                            'Net Int Rate', 'Years', 'Market CAGR', 'NOI Cap Rate Compounded', 
                            'Operation Cashflow', 'Market Cap Appreciation BP', 'IRR', 'Equity Multiple']
        
        first_table = comparison_[['Current Price', 'LTV', 'Loan', 'Equity', '5yrs Swap Rate', 'Spread', 'Net Int Rate']]
        ###format the table
        first_table['Current Price'] = first_table['Current Price'].apply(lambda x: f'${x:,.0f}')
        first_table['Loan'] = first_table['Loan'].apply(lambda x: f'${x:,.0f}')
        first_table['Equity'] = first_table['Equity'].apply(lambda x: f'${x:,.0f}')
        #first_table['LTV'] = first_table['LTV'].apply(lambda x: f'{x:.0%}')
        #first_table['5yrs Swap Rate'] = first_table['5yrs Swap Rate'].apply(lambda x: f'{x:.2%}')
        #first_table['Spread'] = first_table['Spread'].apply(lambda x: f'{x:.2%}')
        #first_table['Net Int Rate'] = first_table['Net Int Rate'].apply(lambda x: f'{x:.2%}')
        first_table = first_table.T

        second_table = comparison_[['Years', 'Market CAGR', 'NOI Cap Rate Compounded', 
                                    'Net Int Rate', 'Operation Cashflow', 'Market Cap Appreciation BP', 'IRR', 'Equity Multiple']]
        second_table['Years'] = second_table['Years'].astype(str) + ' Years'
        second_table['Market CAGR'] = second_table['Market CAGR'].apply(lambda x: f'{x:.2%}')
        second_table['NOI Cap Rate Compounded'] = second_table['NOI Cap Rate Compounded'].apply(lambda x: f'{x:.2%}')
        #second_table['Net Int Rate'] = second_table['Net Int Rate'].apply(lambda x: f'{x:.2%}')
        second_table['Operation Cashflow'] = second_table['Operation Cashflow'].apply(lambda x: f'{x:,.2%}')
        second_table['Market Cap Appreciation BP'] = second_table['Market Cap Appreciation BP'].apply(lambda x: f'{x:.0f}')
        second_table['IRR'] = second_table['IRR'].apply(lambda x: f'{x:.2%}')
        second_table['Equity Multiple'] = second_table['Equity Multiple'].apply(lambda x: f'{x:.2f}')
        second_table = second_table.T

        st.subheader('Loan Comparison')
        st.table(first_table)
        st.subheader('Performance Comparison')
        st.table(second_table)

st.subheader('Individual Search Results', divider= 'blue')
st.write('Summary for :', st.session_state.selected_state, st.session_state.selected_city, st.session_state.selected_slice_city)
############################

summary_filtered_city = summary[(summary['state'] == st.session_state.selected_state) & (summary['city'] == st.session_state.selected_city) & (summary['slice'] == st.session_state.selected_slice_city)].copy()


cashflow_filtered = cashflow[(cashflow['state'] == st.session_state.selected_state) & (cashflow['city'] == st.session_state.selected_city) & (cashflow['slice'] == st.session_state.selected_slice_city)].copy()
cashflow_filtered_show = cashflow_filtered[['price','equity','revenue','debt_payment','loan_payoff','valuation','cashflow']].copy()
for i in cashflow_filtered_show.columns:
    cashflow_filtered_show[i] = cashflow_filtered_show[i].apply(lambda x: f'${x:,.0f}') 
cashflow_filtered_show.columns = ['Price','Equity','Revenue','Debt Payment','Loan Payoff','Valuation','Cashflow']
cashflow_filtered_show.index = cashflow_filtered_show.index.strftime('%Y-%m-%d')


summary_filtered_loan_cond = summary_filtered_city.copy()
summary_filtered_loan_cond['LTV'] = '60%'
summary_filtered_loan_cond['5yrs Swap Rate'] = '3.45%'
summary_filtered_loan_cond['Spread'] = '3.0%'
summary_filtered_loan_cond['Net Int Rate'] = '6.45%'
summary_filtered_loan_cond['Years'] = '5'


summary_filtered_loan_cond = summary_filtered_loan_cond.set_index('market')[['current_price','LTV','loan','equity','5yrs Swap Rate','Spread','Net Int Rate',
                                                                             'market_cagr',
                                                                             'noi_cap_rate_compounded','operation_cashflow','irr','npv','equity_multiple']]
summary_filtered_loan_cond.columns = ['Current Price','LTV','Loan','Equity','5yrs Swap Rate','Spread','Net Int Rate',
                                      'Market CAGR',
                                      'NOI Cap Rate Compounded','Operation Cashflow','IRR','Net Present Value','Equity Multiple']

for col in ['Current Price','Loan','Equity','Net Present Value','Equity Multiple']:
    summary_filtered_loan_cond[col] = summary_filtered_loan_cond[col].apply(lambda x: f'{x:,.2f}')

for col in ['NOI Cap Rate Compounded', 'Operation Cashflow', 'IRR', 'Market CAGR']:
    summary_filtered_loan_cond[col] = summary_filtered_loan_cond[col].apply(lambda x: f'{x:.2%}')

summary_supply_demand = summary_filtered_city.set_index('market')[['demand_vs_supply','demand_yoy_growth','supply_yoy_growth']].copy()
summary_supply_demand.columns = ['Demand vs Supply','Demand YoY Growth','Supply YoY Growth']
summary_supply_demand['Demand vs Supply'] = summary_supply_demand['Demand vs Supply'].apply(lambda x: f'{x:.2%}')
summary_supply_demand['Demand YoY Growth'] = summary_supply_demand['Demand YoY Growth'].apply(lambda x: f'{x:.2%}')
summary_supply_demand['Supply YoY Growth'] = summary_supply_demand['Supply YoY Growth'].apply(lambda x: f'{x:.2%}')


financial_summary = summary_filtered_loan_cond[['NOI Cap Rate Compounded', 'Operation Cashflow', 'Market CAGR','IRR']].copy()



table_filtered = table[(table['state'] == st.session_state.selected_state) & (table['city'] == st.session_state.selected_city) & (table['slice'] == st.session_state.selected_slice_city)].copy()
equilibrium_filtered = equilibrium[(equilibrium['state'] == st.session_state.selected_state) & (equilibrium['city'] == st.session_state.selected_city) & (equilibrium['slice'] == st.session_state.selected_slice_city)].copy()

equlibrium_show = equilibrium_filtered[['absorption_units_12_mo' , 'demand_units', 'net_delivered_units_12_mo', 'inventory_units']].copy()
equlibrium_show.columns = ['Absortion Units 12 Mo', 'Demand Units', 'Net Delivered Units 12 Mo', 'Inventory Units']
equlibrium_show.index = equlibrium_show.index.strftime('%Y-%m-%d')
## todos en porcentaje 
for i in equlibrium_show.columns:
    equlibrium_show[i] = equlibrium_show[i].apply(lambda x: f'{x:,.0f}')


financial_table = table_filtered[['occupancy_rate' ,'market_sale_price_per_unit' , 'market_effective_rentunit',  'market_sale_price_growth', 'market_effective_rent_growth_12_mo',  'market_cap_rate',
                                  'revenue', 'opex','noi']].copy()
financial_table.columns = ['Occupancy Rate','Market Sale Price per Unit','Market Effective Rent per Unit','Market Sale Price Growth','Market Effective Rent Growth 12 Mo','Market Cap Rate',
                            'Revenue','Opex','NOI']
for i in ['Market Sale Price per Unit','Market Effective Rent per Unit','Revenue','NOI']:
    financial_table[i] = financial_table[i].apply(lambda x: f'${x:,.0f}')



for i in ['Market Sale Price Growth','Market Effective Rent Growth 12 Mo','Occupancy Rate','Market Cap Rate', 'Opex']:
    financial_table[i] = financial_table[i].apply(lambda x: f'{x:.2%}')

financial_table.iloc[0 , -2:] = ''


financial_table.index = financial_table.index.strftime('%Y-%m-%d')

ind1 , ind2 = st.columns([2, 1])

with ind1:
    st.subheader('Cashflow Analysis', divider= 'blue')
    st.dataframe(cashflow_filtered_show, use_container_width=True)
with ind2:
    st.subheader('Loan conditions', divider= 'violet')
    st.dataframe(summary_filtered_loan_cond.T, use_container_width=True, height=490)
    
st.markdown('---')
ind_1 , ind_2 = st.columns([1, 2])
with ind_1:
    st.subheader('Supply Demand Forecast ', divider= 'blue')
    ### hace un markdown de colores por cada uno con letras en engrita 
    for i in summary_supply_demand.columns:
        st.metric(i, summary_supply_demand[i].values[0], delta= None, delta_color= 'normal')

with ind_2:
    st.subheader('Supply Demand Data', divider= 'violet')
    st.dataframe(equlibrium_show, use_container_width=True)
    
st.markdown('---')

ind__0 ,ind__1 , ind__2 = st.columns([0.1,2, 0.1])

with ind__1:
    st.subheader('Financial Data', divider= 'green')
    st.dataframe(financial_table, use_container_width=True)


