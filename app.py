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


##3 default all selected
with st.sidebar:
    ###un boton que resetee los filtros
    if st.button('Reset Filters'):
        #st.session_state.selected_states = states
        #st.session_state.slice = classes[0]
        #st.session_state.population = 500000
        st.experimental_rerun()

    slice = st.selectbox('Select Class', options=classes, index=0)
    st.session_state.slice = slice
    box_selector_pop = ['All','+100K', '+500K', '+1M', '+2M' , '+3.5M' , '+5M', '+7.5M', '+10M']
    population = st.selectbox('Select Population', options=box_selector_pop, index=2)
    result = population.replace('+', '').replace('M', '000000').replace('K', '000').replace('All', '0')
    st.session_state.population = result
    st.markdown('--'*20)

with st.expander('States Filter'):
    st.session_state.selected_states = states
    selected_states = st.multiselect('**Select States**', states, default=st.session_state.selected_states)


# Función para asignar colores basados en IRR
def get_color(value, column_name):
    #if value >  column_name.quantile(0.85):
    if value > 25:
        return [0, 128, 0, 200]  # Verde más oscuro
    #elif value > column_name.quantile(0.7):
    elif value > 20:
        return [144, 238, 144, 200]  # Verde super claro
    #elif value > column_name.quantile(0.4):
    elif value > 15:
        # verde limon
        return [173, 200, 47, 200]
    #elif value > column_name.quantile(0.1):
    elif value > 10:
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


summary_filtered = summary[(summary['slice'] == st.session_state.slice) & (summary['state'].isin(selected_states)) & (summary['population'] >= int(st.session_state.population))].copy()
selected_markets = summary_filtered['market'].unique()

if summary_filtered.empty:
    st.error('No data available for the selected filters')
    st.stop()



summary_filtered['IRR_percentage'] = (summary_filtered['irr'] * 100).round(2)
summary_filtered['bar_height'] = summary_filtered['IRR_percentage'] ** 1.7
summary_filtered['color'] = summary_filtered['IRR_percentage'].apply( lambda x: get_color(x, summary_filtered['IRR_percentage']))
summary_filtered['log_population'] = (summary_filtered['population']) ** 0.5
summary_filtered['population_millions'] = (summary_filtered['population'] / 1_000_000).round(2).astype(str) + 'M'

col1 , col2  = st.columns([2, 1])

with col1:
    st.subheader('IRR per US Market + demographic aggregation for ' + st.session_state.slice + ' buildings')
    # Definimos la capa ColumnLayer
    irr_layer = pdk.Layer(
        "ColumnLayer",
        data=summary_filtered,
        get_position=["longitude", "latitude"],
        get_elevation="bar_height",
        elevation_scale=450,  # Ajusta según sea necesario para la visibilidad
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
        zoom=3.5,
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
                    <b>City:</b> {city}<br/>
                    <b>IRR:</b> {IRR_percentage}%<br/>
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
            description="Real Estate IRR per US Market and Population",
            parameters={
                "cull": True,
                "borderRadius": "500px"
                },

            # width= 200,
            # height=500
        ),
        use_container_width=True
        )
    st.markdown('''**Note:** The height of the bars is proportional to the IRR percentage, and the color is based on the IRR percentile.  
                The size of the circles is proportional to the population of the city''')

with col2:
    subcol1, subcol2 = st.columns([0.075, 1])
    with subcol2:
        st.subheader('IRR Summary')
        st.dataframe(summary_filtered[['state', 'city',  'irr', 'population']].sort_values('irr', ascending=False).head(100).set_index(['state', 'city']).assign(irr = lambda x: x['irr'].apply(lambda x: f'{x:.2%}') ,
                                                                                                                        population = lambda x: x['population'].apply(lambda x: f'{x/1000000:,.2f}') + 'M' ),
                                                                                                                        height=600, width=400)  

st.markdown(' ')
st.markdown(' ')

###3 filtros por estado y mercado singulares para examinar por separado
st.header('Individual Market Analysis 🏠', divider= 'blue')

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.5, 0.5]) 




with col1:
    selected_state = st.selectbox('Select State', selected_states, index=0)
    st.session_state.selected_state = selected_state

with col2:
    selected_cities = summary[summary['state'] == selected_state]['city'].unique()
    selected_city = st.selectbox('Select City', selected_cities, index=0)
    st.session_state.selected_city = selected_city

with col3:
    selected_slices_city = summary[(summary['state'] == selected_state) & (summary['city'] == selected_city)]['slice'].unique()
    selected_slice_city = st.selectbox('Select Class', selected_slices_city, index=0, key='slice_city')
    st.session_state.selected_slice_city = selected_slice_city

with col4:
    if st.button('Clear Saved Searches'):
        st.session_state.saved_searches = []
        st.success('All saved searches have been cleared.')

with col5:
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


def filter_summary(states_f, city_f, slice_f):
    return summary[(summary['state'] == states_f) & (summary['city'] == city_f) & (summary['slice'] == slice_f)].copy()

with st.expander('Saved Searches'):
    # if 'saved_searches' in st.session_state:
    #     st.write('Saved Searches:')
    #     st.dataframe(pd.DataFrame(st.session_state.saved_searches).drop_duplicates().T.rename(columns={i : f'Search {i+1}' for i in range(len(st.session_state.saved_searches))}))

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


st.subheader('Individual Search Results')
st.write('Summary for selected market')

#summary_filtered_city = summary[(summary['state'] == st.session_state.selected_state) & (summary['city'] == st.session_state.selected_city) & (summary['slice'] == st.session_state.selected_slice_city)].copy()

#st.write(summary_filtered_city)



