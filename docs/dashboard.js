importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.4/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.4/dist/wheels/panel-0.14.4-py3-none-any.whl', 'pyodide-http==0.1.0', 'google', 'os', 'pandas', 'param', 'pipestonks', 'plotly']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

#!/usr/bin/env python
# coding: utf-8

# In[3]:


import panel as pn
import pandas as pd
import plotly.graph_objects as go

pn.extension('plotly', template='fast')


# In[4]:


from pipestonks.connection.firebase_util import (load_dataframe_from_filename, get_temporary_folder, get_storage_file_format)

from pipestonks.pipeline.workflow import (get_target_stocks, get_filtered_data)

from google.cloud import storage

import os

root_folder = ""
data_folder = os.path.join(root_folder, "br_stock_exchange/")
reports_folder = os.path.join(root_folder, "reports/")
temp_folder = get_temporary_folder()
output_file_format = get_storage_file_format()

stocks_to_filter = get_target_stocks()


# In[5]:


client = storage.Client(project="pipestonks")
bucket = client.bucket("pipestonks.appspot.com")

list_objects = bucket.list_blobs(prefix=data_folder)
filtred_info = get_filtered_data(list_objects, stocks_to_filter)


# ### Interact
# 
# In the \`\`interact\`\` model the widgets are automatically generated from the arguments to the function or by providing additional hints to the \`\`interact\`\` call. This is a very convenient way to generate a simple app, particularly when first exploring some data.  However, because widgets are created implicitly based on introspecting the code, it is difficult to see how to modify the behavior.  Also, to compose the different components in a custom way it is necessary to unpack the layout returned by the \`\`interact\`\` call, as we do here:

# ### Reactive
# 
# The reactive programming model is similar to the \`\`interact\`\` function but relies on the user (a) explicitly instantiating widgets, (b) declaring how those widgets relate to the function arguments (using the \`\`bind\`\` function), and (c) laying out the widgets and other components explicitly. In principle we could reuse the \`\`get_plot\`\` function from above here but for clarity we will repeat it:

# In[12]:


stocks = pn.widgets.Select(name='Stocks', options=stocks_to_filter)
window = pn.widgets.IntSlider(name='Window Size', value=6, start=1, end=21)

def get_df(ticker, window_size):
    df = filtred_info[ticker][1]
    df['Date'] = pd.to_datetime(df.index)
    return df.set_index('Date').rolling(window=window_size).mean().reset_index()

def get_plot(ticker, window_size):
    df = get_df(ticker, window_size)
    return go.Scatter(x=df.Date, y=df.Close)

#pn.Row(
#    pn.Column("Pipestonks TM", stock, window),
#    pn.bind(get_plot, ticker, window),
#    sizing_mode='stretch_width'
#)


# ### Parameterized class
# 
# Another approach expresses the app entirely as a single \`\`Parameterized\`\` class with parameters to declare the inputs, rather than explicit widgets. The parameters are independent of any GUI code, which can be important for maintaining large codebases, with parameters and functionality defined separately from any GUI or panel code. Once again the \`\`depends\`\` decorator is used to express the dependencies, but in this case the dependencies are expressed as strings referencing class parameters, not parameters of widgets. The parameters and the \`\`plot\`\` method can then be laid out independently, with Panel used only for this very last step.

# In[13]:


import param

class StockExplorer(param.Parameterized):
    stock = param.Selector(default='PETR4', objects=stocks_to_filter)

    window_size = param.Integer(default=6, bounds=(1, 21))

    @param.depends('stock', 'window_size')
    def plot(self):
        return get_plot(self.stock, self.window_size)

explorer = StockExplorer()

pn.Row(
    pn.Column(explorer.param),
    explorer.plot
)


# ### Callbacks
# 
# The above approaches are all reactive in some way, triggering actions whenever manipulating a widget causes a parameter to change, without users writing code to trigger callbacks explicitly.  Explicit callbacks allow complete low-level control of precisely how the different components of the app are updated, but they can quickly become unmaintainable because the complexity increases dramatically as more callbacks are added. The approach works by defining callbacks using the \`\`.param.watch\`\` API that either update or replace the already rendered components when a watched parameter changes:

# In[14]:


stock = pn.widgets.Select(
    name='Stock', options=stocks_to_filter
)
window = pn.widgets.IntSlider(
    name='Window', value=6, start=1, end=21
)

row = pn.Row(
    pn.Column("Pipestonks TM", stock, window, sizing_mode="fixed", width=300),
    get_plot(stock.options[0], window.value)
)

def update(event):
    row[1].object = get_plot(stock.value, window.value)

stock.param.watch(update, 'value')
window.param.watch(update, 'value')

row


# In practice, different projects will be suited to one or the other of these APIs, and most of Panel's functionality should be available from any API.

# ## App
# 
# This notebook may also be served as a standalone application by running it with \`panel serve stocks_plotly.ipynb\`. Above we enabled a custom \`template\`, in this section we will add components to the template with the \`.servable\` method:

# In[16]:


stock.servable(area='sidebar')
window.servable(area='sidebar')

pn.panel(pn.bind(get_plot, stock, window)).servable(title='PipeStonks TM');



await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()