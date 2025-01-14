#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/20/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import queue
import threading
import base64
import io
import time
from typing import Any, Optional, NamedTuple

import numpy as np
from PIL import Image

import dash
import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import jacinle


def create_visualizer_app(title, tab_queue_descriptions):
    # Initialize the Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    elements = [html.H1(title)]

    tabs = list()
    for tab_name, queue_descriptions in tab_queue_descriptions.items():
        graph_elements = list()
        groups = dict()
        for name in queue_descriptions:
            if queue_descriptions[name].attach_to != '':
                continue
            data_type = queue_descriptions[name].data_type
            group = queue_descriptions[name].group

            data_element = None
            if data_type == 'float':
                data_element = dcc.Graph(id=f'{name}-line-plot', style={'width': '100%'})
            elif data_type == 'image':
                data_element = html.Div([
                    html.Img(id=f'{name}-img', style={'width': '100%'}),
                    html.Div(id=f'{name}-img-update')
                ])
            else:
                raise ValueError(f'Unknown data type: {data_type}')

            if group == '':
                graph_elements.append(html.Div([
                    html.H2(name),
                    data_element
                ]))
            else:
                if group not in groups:
                    groups[group] = list()
                    graph_elements.append(groups[group])

                kwargs = dict()
                width = queue_descriptions[name].width_in_group
                if width > 0:
                    kwargs['width'] = width

                groups[group].append(dbc.Col(html.Div([
                    html.H2(name),
                    data_element
                ]), **kwargs))

        for i, elem in enumerate(graph_elements):
            if isinstance(elem, list):
                graph_elements[i] = html.Div([
                    dbc.Row(elem)
                ])
        tabs.append(dbc.Card(dbc.CardBody(graph_elements)))

    elements.append(dbc.Tabs([dbc.Tab(tab, label=tab_name) for tab_name, tab in zip(tab_queue_descriptions.keys(), tabs)]))
    elements.extend([
        dcc.Interval(
            id='interval-component-plot',
            interval=1000,  # Update every second
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-component-img',
            interval=5000,  # Update every second
            n_intervals=0
        )
    ])

    app.layout = html.Div(elements, style={'margin': '20px'})
    return app


class QueueDescription(NamedTuple):
    name: str
    data_type: str
    max_datapoints: int = 200
    group: str = ''
    width_in_group: int = 0
    attach_to: str = ''


class QueueItem(NamedTuple):
    timestamp: float
    data: Any


class RobotStateVisualizer(object):
    def __init__(self, title: str):
        self.app = None
        self.title = title
        self.queues = dict()
        self.queue_descriptions = dict()
        self.name_to_tab = dict()
        self.queues_updated = dict()
        self.mutex = threading.Lock()
        self.init_timestamp = time.time()

        self.main_thread = None

    def mainloop(self):
        self.app = create_visualizer_app(self.title, self.queue_descriptions)

        outputs_plots = list()
        outputs_imgs = list()
        for (tab_name, name), queue in self.queues.items():
            desc = self.queue_descriptions[tab_name][name]
            if desc.attach_to != '':
                continue
            data_type = desc.data_type
            if data_type == 'float':
                outputs_plots.append(Output(f'{name}-line-plot', 'figure'))
            elif data_type == 'image':
                outputs_imgs.append(Output(f'{name}-img', 'src'))
                outputs_imgs.append(Output(f'{name}-img-update', 'children'))

        self.app.callback(
            outputs_plots,
            [Input('interval-component-plot', 'n_intervals')]
        )(self.update_graph_live)
        self.app.callback(
            outputs_imgs,
            [Input('interval-component-img', 'n_intervals')]
        )(self.update_img_live)

        # Do not print the debug message
        self.app.run_server(host="0.0.0.0", threaded=True, debug=True, use_reloader=False)

    def start(self):
        # Run it in a separate thread
        self.main_thread = threading.Thread(target=self.mainloop, daemon=True)
        self.main_thread.start()

    def reset(self, title: Optional[str] = None, queues: Optional[dict] = None):
        if title is not None:
            self.title = title

        with self.mutex:
            self.queues = dict()
            self.queue_descriptions = dict()
            self.name_to_tab = dict()
            self.queues_updated = dict()
            self.init_timestamp = time.time()

            if queues is not None:
                for (tab, name), desc in queues.items():
                    self.queues[(tab, name)] = list()
                    self.name_to_tab[name] = tab
                    if tab not in self.queue_descriptions:
                        self.queue_descriptions[tab] = dict()
                    self.queue_descriptions[tab][name] = desc
                    self.queues_updated[(tab, name)] = False

    def register_queue(self, tab, name, data_type: str, max_datapoints: int = 200, group: str = '', **kwargs):
        self.queues[(tab, name)] = list()
        self.name_to_tab[name] = tab
        if tab not in self.queue_descriptions:
            self.queue_descriptions[tab] = dict()
        self.queue_descriptions[tab][name] = QueueDescription(name, data_type, max_datapoints=max_datapoints, group=group, **kwargs)
        self.queues_updated[(tab, name)] = False

    def update_queue(self, name, timestamp, data, tab=None):
        if tab is None:
            tab = self.name_to_tab[name]
        with self.mutex:
            queue = self.queues[(tab, name)]
            queue.append(QueueItem(timestamp, data))
            max_datapoints = self.queue_descriptions[tab][name].max_datapoints
            if len(queue) > max_datapoints:
                self.queues[(tab, name)] = queue[-max_datapoints:]
            self.queues_updated[(tab, name)] = True

    def update_queue_with_mutex(self, name, timestamp, data, tab=None):
        if tab is None:
            tab = self.name_to_tab[name]

        queue = self.queues[(tab, name)]
        queue.append(QueueItem(timestamp, data))
        max_datapoints = self.queue_descriptions[tab][name].max_datapoints
        if len(queue) > max_datapoints:
            self.queues[(tab, name)] = queue[-max_datapoints:]
        self.queues_updated[(tab, name)] = True


    def update_graph_live(self, n):
        figures = list()

        with self.mutex:
            for (tab_name, queue_name), queue in self.queues.items():
                description = self.queue_descriptions[tab_name][queue_name]
                if description.attach_to != '':
                    continue

                if description.data_type == 'float':
                    x_data = [item.timestamp - self.init_timestamp for item in queue]
                    y_data = [item.data for item in queue]

                    data = [
                        go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=queue_name)
                    ] + self._get_attached_queue_data(tab_name, queue_name)
                    # print(f'Queue {queue_name} updated. {len(x_data)} points. {len(data)} data series.')

                    line_fig = go.Figure(data, layout=go.Layout(
                        xaxis=dict(range=[x_data[0], x_data[-1] + 5]),
                        yaxis=dict(range=[min(y_data), max(y_data)])
                    ))
                    # Make the layout more compact
                    line_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    figures.append(line_fig)
                else:
                    continue

        return tuple(figures)

    def _get_attached_queue_data(self, tab_name, queue_name):
        additional_data = list()
        for (attached_tab_name, attached_queue_name), queue in self.queues.items():
            description = self.queue_descriptions[attached_tab_name][attached_queue_name]
            if attached_tab_name == tab_name and description.attach_to == queue_name:
                queue = self.queues[(attached_tab_name, attached_queue_name)]
                additional_data.append(
                    go.Scatter(
                        x=[item.timestamp - self.init_timestamp for item in queue],
                        y=[item.data for item in queue],
                        mode='lines+markers',
                        name=attached_queue_name
                    )
                )
        return additional_data

    def update_img_live(self, n):
        figures = list()
        with self.mutex:
            for (tab_name, queue_name), queue in self.queues.items():
                description = self.queue_descriptions[tab_name][queue_name]

                if description.data_type == 'image':
                    if len(queue) == 0:
                        figures.append('')
                        figures.append('')
                        continue
                    if not self.queues_updated[(tab_name, queue_name)]:
                        figures.append(dash.no_update)
                        figures.append(dash.no_update)
                        continue

                    timestamp, img = queue[-1].timestamp, queue[-1].data
                    img = Image.fromarray(img)
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    img_src = f'data:image/png;base64,{img_data}'
                    figures.append(img_src)
                    figures.append(time.strftime('Last updated: %Y-%m-%d %H:%M:%S', time.localtime(timestamp)))
                    self.queues_updated[(tab_name, queue_name)] = False
                else:
                    continue

        if len(figures) == 0:
            return
        return tuple(figures)

