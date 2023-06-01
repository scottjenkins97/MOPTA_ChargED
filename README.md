### Planning Electric Vehicle Charging Stations

This interactive interface is designed to allow stakeholders to run different scenarios, visualize results, and make infrastructure decisions.

You can navigate between pages in the side bar on the left. You are currently on the 'Home' page. The 'Optimize' page....

The sensitivity of different parameter settings, model configurations, and scenarios, can be further explored in an interactive front-end we have built in addition to our solver, using the Python library Streamlit. On opening the application, the user is presented with a map of the \glspl{ev} locations. The \enquote{Optimize} button calls our solver, and after each iteration of Algorithm \ref{alg:solve}, the map, along with metric boxes and additional plots, is updated. In the left-hand side-bar, the user can select different solver parameters, fix the number of \glspl{cs} to be built, or provide a new data-set of \gls{ev} locations. Once the optimization run is complete, the \enquote{Validate} button tests the solution for robustness on a specified number of new scenarios, and plots the results. We include screenshots of our interface in Appendix \ref{a:streamlit}.
