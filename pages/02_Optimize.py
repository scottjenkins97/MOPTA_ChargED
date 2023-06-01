import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from utilities.mopta_solver import MOPTASolver
import logging

## Set page config must be first command
st.set_page_config(page_title='ChargED - Optimiser',
                   page_icon=':car:',
                   layout='wide',  ## 'centered','wide'
                   initial_sidebar_state= 'auto' # 'expanded'  ## 'auto','collapsed','expanded'
                   )
st.markdown('#### Building Charging Infrastructure to Minimise Total Cost')

## Load Default Car Locations
cwd = os.getcwd()
file = 'full_car_locations.csv'   # full_car_locations.csv / small_car_location.csv
if 'car_locations' not in st.session_state:
    st.session_state['car_locations'] = pd.read_csv(os.path.join(cwd, 'locations', file))

## Initialise Session State

if 'fixed_station_number' not in st.session_state:
    st.session_state['fixed_station_number'] = None
if 'number_of_samples' not in st.session_state:
    st.session_state['number_of_samples'] = 3

if 'optimise_status' not in st.session_state:
    st.session_state['optimise_status'] = 0         # 0: not optimised, 1: in progress, 2: finished
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = ''
if 'solution' not in st.session_state:
    st.session_state['solution'] = 0
if 'L_sol' not in st.session_state:
    st.session_state['L_sol'] = 0
if 'n_sol' not in st.session_state:
    st.session_state['n_sol'] = 0

## Metrics
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = '-'
if 'drive_charge_cost' not in st.session_state:
    st.session_state['drive_charge_cost'] = '-'
if 'locations_built' not in st.session_state:
    st.session_state['locations_built'] = '-'
if 'chargers_built' not in st.session_state:
    st.session_state['chargers_built'] = '-'

if 'validate_df' not in st.session_state:
    st.session_state['validate_df'] = 0
if 'total_cost_fig' not in st.session_state:
    st.session_state['total_cost_fig'] = ''


map_cont = st.empty()

###########################
## Sidebar
###########################

# Own Data Selection
own_data = st.sidebar.checkbox('Import Vehicle Locations?')  # Tick if want to upload data
if own_data:
    uploaded_file = st.sidebar.file_uploader('Upload your .csv file here (same format as default file, with "x" and "y" columns only, 1 row per vehicle)', accept_multiple_files=False)
    if uploaded_file is not None:
        car_locations = pd.read_csv(uploaded_file, delimiter=';', names=['x', 'y'])
        st.session_state['car_locations'] = car_locations
        map_cont.empty()
        with map_cont.container():
            vehicle_locations = go.Scatter(x=car_locations.iloc[:, 0], y=car_locations.iloc[:, 1], mode='markers',name='Vehicle Locations', marker=dict(size=2, color='blue'))
            x_axis = go.XAxis(title='x coordinate / Miles', showgrid=False, mirror=True, ticks='outside', showline=True)
            y_axis = go.YAxis(title='y coordinate / Miles', showgrid=False, mirror=True, ticks='outside', showline=True)
            layout = go.Layout(title_text='Vehicle Locations',xaxis=x_axis, yaxis=y_axis, height=500)  #title_x=0.5,
            start_fig = go.Figure(data=[vehicle_locations], layout=layout)
            st.plotly_chart(start_fig, use_container_width=True)
else:
    car_locations  = pd.read_csv(os.path.join(cwd, 'locations', file))
    st.session_state['car_locations'] = car_locations
    map_cont.empty()
    with map_cont.container():
        vehicle_locations = go.Scatter(x=car_locations.iloc[:, 0], y=car_locations.iloc[:, 1], mode='markers', name='Vehicle Locations', marker=dict(size=2,color='blue'))
        x_axis = go.XAxis(title='x coordinate / Miles',showgrid=False,mirror=True,ticks='outside',showline=True)
        y_axis = go.YAxis(title='y coordinate / Miles',showgrid=False,mirror=True,ticks='outside',showline=True)
        layout = go.Layout(title_text='Vehicle Locations',xaxis=x_axis,yaxis=y_axis,height=500)  # title_x=0.5
        start_fig = go.Figure(data=[vehicle_locations], layout=layout)
        st.plotly_chart(start_fig, use_container_width=True)

# Specify the total number of stations
fixed_chargers = st.sidebar.checkbox('Specify Total Stations Built?',
                                     help='Check this box if you want to specify the number of stations built'
                                     )
if fixed_chargers:
    fixed_station_number = st.sidebar.slider('How many stations should be built?',
                                              min_value=1,
                                              max_value=len(car_locations),
                                              step=1,
                                              disabled=not fixed_chargers,
                                              value=600)
    st.session_state['fixed_station_number'] = fixed_station_number   ## Add to session state

# Vary upper bound of number of stations at each location
station_ub = st.sidebar.slider('Maximum Stations at any given Location:',
                               min_value=1,
                               max_value=20,
                               step=1,
                               value=8)

# Vary the Cost & Service Level Parameters used by the solver
c_b = st.sidebar.number_input('Annualised Construction Cost (Per Location)', min_value=0, value=5000)
c_m = st.sidebar.number_input('Annualised Maintenence Cost (Per Station)', min_value=0, value=500)
c_d = st.sidebar.number_input('Drive cost per mile', min_value=0.0, value=.041, format='%0.5f')
c_c = st.sidebar.number_input('Charge cost per mile', min_value=0.0, value=0.0388, format='%0.5f')
service_level = st.sidebar.number_input('Breakdown SLA (%)',
                                        min_value=0,
                                        value=95,
                                        help='The minimal percentage of vehicles which must reach a charger')
st.session_state['service_level'] = service_level/100

st.sidebar.text('Advanced Settings')
## ADVANCED SETTINGS
time_limit = st.sidebar.number_input('Time Limit for Solver to Improve Solution',
                                     min_value=0,
                                     value=300,
                                     help = 'Maximum time in seconds for solver to search for improved solution before terminates (Advanced Setting)')

epsilon_stable = st.sidebar.number_input('Minimum required Improvement between Iterations ($)',
                                     min_value=0,
                                     value=10000,
                                     help = 'The solver will terminate after an iteration if it does not improve the total cost by more than this amount (Advanced Setting)')

st.session_state['number_of_samples'] = st.sidebar.number_input('Number of Samples Optimised Over',
                                                                min_value = 1,
                                                                value = 3,
                                                                help='Optimising over a greater number of samples can take longer, but produce a more robust solution. (Advanced Setting)')
####################
## End Sidebar
####################

## METRIC CONTAINER
metric_container = st.container()
metric_container.write("Metrics from the latest solve iteration")
col1, col2, col3, col4 = metric_container.columns(4)
total_cost_metric = col1.metric("Total Cost", st.session_state['total_cost'])
drive_charge_cost_metric = col2.metric("Drive & Charge Cost", st.session_state['drive_charge_cost'])
built_locs_metric = col3.metric("Locations Built", st.session_state['locations_built'])
built_chargers_metric = col4.metric("Chargers Built", st.session_state['chargers_built'])

# CONTAINERS TO HOLD THE OTHER CHARTS
total_cost_cont = st.empty()
other_cost_cont = st.empty()
charger_count_cont = st.empty()

# Initialise lists to store metric values after each iteration
total_cost_iter = []; drive_charge_cost_iter = []
chargers_built_iter = []; locations_built_iter = []

##################################

## MOPTA SOLVER UPDATES VARIABLES AFTER EACH ITERATION
def streamlit_update(solver):
    # Extract results from latest iteration of solver
    L = solver.L   # Contains all locations considered
    L_new = solver.added_locations[-1]  # [-1] for locations added in most recent iteration
    b, n, u = solver.solutions[-1]  # b = build binary vars, n = number of chargers built, u = allocation matrix

    # Create dataframe for easier plotting
    results_df = pd.DataFrame(L, columns=['loc_x', 'loc_y'])
    results_df['b'] = b
    results_df['n'] = n
    # st.write(results_df)
    st.session_state['results_df'] = results_df

    #############################
    ## UPDATE SCATTER WITH CHARGER LOCATIONS
    map_cont.empty()
    with map_cont.container():
        vehicle_locations = go.Scatter(x=car_locations.iloc[:, 0],y=car_locations.iloc[:, 1],mode='markers',name='Vehicle Locations',marker=dict(size=2,color='blue'))
        built_locations = go.Scatter(x=results_df[results_df['b'] == 1].loc_x, y=results_df[results_df['b'] == 1].loc_y,mode='markers',name='Built Locations',
                                     marker=dict(size=results_df[results_df['b'] == 1].n,color='red'))
        considered_locations = go.Scatter(x=results_df[results_df['b'] == 0].loc_x, y=results_df[results_df['b'] == 0].loc_y,mode='markers',name='Considered Locations (Not Built)',
                                          marker=dict(size=2,color='black',symbol='x-open'))
        x_axis = go.layout.XAxis(title='x coordinate / Miles',showgrid=False,mirror=True,ticks='outside',showline=True) # go.XAxis is deprecated
        y_axis = go.layout.YAxis(title='y coordinate / Miles',showgrid=False,mirror=True,ticks='outside',showline=True)
        layout = go.Layout(title_text='Vehicle & Charger Locations (Built and Considered)',xaxis=x_axis,yaxis=y_axis,height= 500)   #title_x=0.5
        map_fig = go.Figure(data=[vehicle_locations, built_locations, considered_locations], layout=layout)
        st.plotly_chart(map_fig, use_container_width=True)


    #################################
    ## METRIC PANEL UPDATE
    locations_built = results_df['b'].sum();locations_built_iter.append(locations_built)
    chargers_built = results_df['n'].sum(); chargers_built_iter.append(chargers_built)
    st.session_state['locations_built'] = locations_built
    st.session_state['chargers_built'] = chargers_built

    # Total Cost KPI
    total_cost = solver.m.kpi_value_by_name(name='total_cost')  ## 'build_cost', 'maintenance_cost', 'drive_charge_cost'
    st.session_state['total_cost'] = total_cost
    total_cost_iter.append(total_cost)
    try:
        total_cost_improvement = total_cost_iter[-1] - total_cost_iter[-2]
    except:
        total_cost_improvement = 0

    # Drive-Charge Cost KPI
    drive_charge_cost = solver.m.kpi_value_by_name(name='drive_charge_cost')
    st.session_state['drive_charge_cost'] = drive_charge_cost
    drive_charge_cost_iter.append(drive_charge_cost)

    # Update Metric Panels
    total_cost_metric.metric("Total Cost",
                             "$" + str(int(st.session_state['total_cost'])),
                             "$" + str(np.round(total_cost_improvement,0))) # delta_color="inverse"
    drive_charge_cost_metric.metric("Drive Charge Cost",
                                        "$" + str(int(st.session_state['drive_charge_cost'])))
    built_locs_metric.metric("Locations Built", st.session_state['locations_built'])
    built_chargers_metric.metric("Chargers Built",st.session_state['chargers_built'])

    ########################################################
    ## TOTAL COST LINE PLOT
    total_cost_cont.empty()                     # Use .empty() to ensure we overwrite figure
    with total_cost_cont.container():
        total_cost_fig = px.line(total_cost_iter, template='simple_white', markers='x')
        total_cost_fig.update_xaxes(title='Iteration', showgrid=False, mirror=True, ticks='outside', showline=True,
                                    tickmode='linear', tick0=1, dtick=1)
        total_cost_fig.update_yaxes(title='Total Cost /$', showgrid=False, mirror=True, ticks='outside', showline=True)
        total_cost_fig.update_layout(title_text='Total Cost / $', title_x=0.5, showlegend=False)
        st.session_state['total_cost_fig'] = total_cost_fig
        st.plotly_chart(st.session_state['total_cost_fig'])

    ## DRIVE-CHARGE COST LINE PLOT
    other_cost_cont.empty()  # Use .empty() to ensure we overwrite figure
    with other_cost_cont.container():
        other_cost_fig = px.line(drive_charge_cost_iter, template='simple_white', markers='x')
        other_cost_fig.update_xaxes(title='Iteration', showgrid=False, mirror=True, ticks='outside', showline=True,
                                    tickmode='linear', tick0=1, dtick=1)
        other_cost_fig.update_yaxes(title='Drive-Charge Cost /$', showgrid=False, mirror=True, ticks='outside', showline=True)
        other_cost_fig.update_layout(title_text='Drive-Charge Cost / $', title_x=0.5, showlegend=False)
        other_cost_fig = st.plotly_chart(other_cost_fig)

    ## CHARGER DISTRIBUTION BAR CHART
    charger_count_cont.empty()
    with charger_count_cont.container():
        charger_counts = results_df[results_df['b']==1]['n'].value_counts()
        charger_count_fig = px.bar(charger_counts, template='simple_white')
        charger_count_fig.update_xaxes(title='Chargers at Location', showgrid=False, mirror=True, tickvals = [1, 2, 3, 4, 5, 6, 7, 8])
        charger_count_fig.update_yaxes(title='Number of Locations', showgrid=False, mirror=True, showline=True)
        charger_count_fig.update_layout(title_text='Number of Chargers by Location', title_x=0.5, showlegend=False)
        charger_count_fig = st.plotly_chart(charger_count_fig)

## END OF SOLVER UPDATE FUNCTION


## Define mopta_solver (outside the optimise button)
mopta_solver = MOPTASolver(car_locations=car_locations.to_numpy(),
                           loglevel=logging.INFO,  # logging.INFO  logging.DEBUG
                           build_cost=c_b,
                           maintenance_cost=c_m,
                           drive_cost=c_d,
                           charge_cost=c_c,
                           service_level=service_level / 100,  # divide by 100 to ensure between 0 and 1
                           station_ub=station_ub,
                           fixed_station_number = st.session_state['fixed_station_number'],
                           streamlit_callback=streamlit_update)  # We set streamlit_update as the callback function



#######################
## Optimise Button

if st.button('Optimise...', help='Start the optimiser with the parameters selected in the sidebar'):
    st.session_state['optimise_status'] = 1   # Optimisation in progress
    # Add Initial Locations
    if st.session_state['fixed_station_number'] is None:
        number_of_initial_locations = int(np.ceil(len(car_locations) / 10))
    else:
        number_of_initial_locations = int(max(np.ceil(len(car_locations)/10),st.session_state['fixed_station_number']))# 400
    mopta_solver.add_initial_locations(number_of_initial_locations, mode='k-means', seed=0)  # mode='random'
    # Add User-Specified number of Samples
    mopta_solver.add_samples(num=st.session_state['number_of_samples'])
    # Call the solve function
    n, L = mopta_solver.solve(verbose=False, timelimit=time_limit, epsilon_stable=epsilon_stable)
    # Update Session State with n and L
    st.session_state['n_sol'] = n; st.session_state['L_sol'] = L

    # Create dataframe of all solutions by iteration
    solution_iterations = mopta_solver.solutions
    location_iterations = mopta_solver.added_locations
    iterations = len(solution_iterations)  # The number of iterations completed overall
    ## First Iteration: [0] index
    sol_df = pd.DataFrame(location_iterations[0], columns=['loc_x','loc_y'])
    sol_df['b'] = solution_iterations[0][0]
    sol_df['n'] = solution_iterations[0][1]
    sol_df['iteration'] = 1
    ## Append Data From Subsequent Iterations
    for i in range(2,iterations+1):
        location_data = pd.DataFrame(location_iterations[0])  # start with the initial locations
        for j in range(1,i):
            location_data = pd.concat([location_data,pd.DataFrame(location_iterations[j])],axis=0)
        iter_df = pd.DataFrame(location_data).rename({0:'loc_x',1:'loc_y'},axis=1)
        iter_df['b'] = solution_iterations[i-1][0]
        iter_df['n'] = solution_iterations[i-1][1]
        iter_df['iteration'] = i
        sol_df = pd.concat([sol_df,iter_df])

    # Write dataframe of solutions to csv, and store in session state
    sol_df.to_csv('sol_df.csv'); st.session_state['solution'] = sol_df
    st.write(f'Finished Solving after {iterations} Iterations.Solution exported as "sol_df.csv" file.')

    # Update session state: optimisation button has been clicked
    st.session_state['optimise_status'] = 2

## END OF OPTIMISE BUTTON


## FINAL SOLUTION: METRICS AND ANIMATED PLOT
if st.session_state['optimise_status'] == 2:
    total_cost_metric.metric("Total Cost","$" + str(int(st.session_state['total_cost'])))
    drive_charge_cost_metric.metric("Drive Charge Cost","$" + str(int(st.session_state['drive_charge_cost'])))
    built_locs_metric.metric("Locations Built", st.session_state['locations_built'])
    built_chargers_metric.metric("Chargers Built", st.session_state['chargers_built'])

    map_cont.empty()
    with map_cont.container():
        sol_df = st.session_state['solution']
        sol_df['b'] = sol_df['b'].astype(str)
        built_df = sol_df[sol_df['b']=='1']
        final_fig = px.scatter(built_df,
                               x='loc_x',
                               y='loc_y',
                               animation_frame='iteration',
                               title='Final Infrastructure Plan (Small Circle: Vehicle Locations, Large Circle: Built Locations)',
                               height=600
                               )
        final_fig.update_xaxes(title='x coordinate/Miles', showgrid=False, mirror=True, ticks='outside', showline=True)
        final_fig.update_yaxes(title='y coordinate/Miles', showgrid=False, mirror=True, ticks='outside', showline=True)
        vehicle_locations = go.Scatter(x=car_locations.iloc[:, 0], y=car_locations.iloc[:, 1], mode='markers',
                                       name='Vehicle Locations', marker=dict(size=2, color='red', symbol='diamond'))
        final_fig.add_trace(vehicle_locations)
        try:
            final_fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000   # Change animation speed
        except:
            pass
        st.plotly_chart(final_fig, use_container_width=True)


## VALIDATE SOLUTION BUTTON
validate_iterations = st.number_input('Once Optimised, test solution against a number of unseen samples, specified below', min_value=1, max_value=None, value=100)

if st.button('Validate Solution',help='Click to test the solution with unseen samples to check robustness'):
    if st.session_state['optimise_status'] != 2:
        st.write('No solution yet generated. Please click the "Optimise" button to solve for your selected parameters')
    else:
        st.write(f'Testing Found Solution on {validate_iterations} Unseen Vehicle Samples.')
        objective_values, service_levels, mip_gaps = mopta_solver.allocation_problem(n_iter=validate_iterations,
                                                                                     L_sol=st.session_state['L_sol'],
                                                                                     n_sol=st.session_state['n_sol'],
                                                                                     verbose=False)
        # Add validation results to session state
        validate_df = pd.DataFrame({'objective':objective_values,'service_level':service_levels,'mip_gap':mip_gaps})
        st.session_state['validate_df'] = validate_df

        feasible = st.session_state['validate_df'][st.session_state['validate_df']['service_level']==st.session_state['service_level']].reset_index(drop=True)
        infeasible = st.session_state['validate_df'][st.session_state['validate_df']['service_level']<st.session_state['service_level']].reset_index(drop=True)
        feasible_count = len(feasible)

        st.write(f"Of the {validate_iterations} samples tested, {feasible_count} were 'feasible', that is an allocation was found sending 95% of vehicles which needed to charge to a charge station. " )
        validate_col1,validate_col2 = st.columns(2)
        with validate_col1:
            feasible_fig = px.histogram(feasible['objective'], template='simple_white')
            feasible_fig.update_xaxes(title='Total Cost', showgrid=False, mirror=True)
            feasible_fig.update_yaxes(title='Frequency Density', showgrid=False, mirror=True, showline=True)
            feasible_fig.update_layout(title_text=f'Histogram of Total Cost for {feasible_count} Feasible Solutions', title_x=0, showlegend=False)
            st.plotly_chart(feasible_fig, use_container_width=True)
        with validate_col2:
            infeasible_fig = px.scatter(infeasible,x='service_level',y='objective',template='simple_white')
            infeasible_fig.update_xaxes(title='Service Level',showgrid=False,mirror=True)
            infeasible_fig.update_yaxes(title='Total Cost',showgrid=False,mirror=True)
            infeasible_fig.update_layout(title_text=f'Service Level vs. Total Cost for {validate_iterations - feasible_count} Infeasible Solutions',
                                         title_x=0, showlegend=False)
            st.plotly_chart(infeasible_fig,use_container_width=True)

## END OF VALIDATE FUNCTION
########################################################################
## END OF PAGE