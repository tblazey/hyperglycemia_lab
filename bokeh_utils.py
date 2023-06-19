#!/usr/bin/python

#Load libs
from bokeh.core.validation import silence
from bokeh.io import output_notebook
from bokeh.core.validation.warnings import MISSING_RENDERERS
from bokeh.events import PointEvent, Tap
from bokeh.layouts import row, layout, column
from bokeh.models import CrosshairTool, ColorBar, ColumnDataSource, InlineStyleSheet, LinearColorMapper, Range1d, Slider, Spacer
from bokeh.palettes import all_palettes, interp_palette
from bokeh.plotting import figure
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.stats as stats

#Ignore warning that says we have some missing renders...s
silence(MISSING_RENDERERS, True)

#Information about orienation/axes
views = ['sag', 'tra', 'cor']
view_info = {'sag':[['y', 'z'], 'x', 0],
             'tra':[['x', 'y'], 'z', 2],
             'cor':[['x', 'z'], 'y', 1]}
             

def overlay_plot(under_source, over_sources, under_data, over_data, 
                 over_idx, ori, cmaps, coord, cross, dims, title=None):
            
    #Use orientation information to setup dimensions
    x = view_info[ori][0][0]
    y = view_info[ori][0][1]
    dw = dims[x]
    dh = dims[y]
    h_x = f"anat_{x}"
    v_y = f"anat_{y}"
    
    #Ploting options based on orientation
    if ori == 'tra':
        x_range = Range1d(dw, 0, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'black'
    elif ori == 'sag':
        x_range = Range1d(0, dw, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'white'
    else:
        x_range = Range1d(dw, 0, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'white'
    
    #Create plot
    p = figure(x_range=x_range, y_range=y_range, name='image')
    p.image(ori, source=under_source, x=0, y=0, dw=dw, dh=dh,
            level="image", color_mapper=cmaps[0])
    p.image(ori, source=over_sources[over_idx], x=0, y=0, dw=dw, dh=dh,
            level="image", color_mapper=cmaps[1], alpha=0.8)
    p.line(x=h_x, y=view_info[ori][0][1], line_color="black", line_width=6,
           line_alpha=0.6, line_dash='dashed', source=cross)
    p.line(x=view_info[ori][0][0], y=v_y, line_color="black", line_width=6,
           line_alpha=0.6, line_dash='dashed', source=cross)
    
    #Function for handling mouse events
    mouse_func = mouse_wrapper(ori, [under_source] + over_sources, [under_data] + over_data, coord, cross)
    p.on_event('tap', mouse_func)
    
    #Plot styling
    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_color= None
    p.toolbar_location = None
    if title is not None:
        p.title = title
        p.title.text_font_style = "bold"
        p.title.text_font_size = "42px"
        p.title.align = 'center'
        p.title.text_color = title_color
    
    return p

def img_kde(img_1, img_2):
    
    #Make grid to estimate pdf on
    min_1 = img_1.min()
    max_1 = img_1.max()
    min_2 = img_2.min()
    max_2 = img_2.max()
    X, Y = np.mgrid[min_1:max_1:100j, min_2:max_2:100j]

    #Estimate kd
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([img_1, img_2])
    kernel = stats.gaussian_kde(values)
    den = kernel(positions).reshape(X.shape)

    #Interpolate image values onto pdf
    return interp.interpn([X[:, 0], Y[0, :]], den, np.stack((img_1, img_2)).T,
                          bounds_error=False, fill_value=0.0)

def create_colorbar(c_map, unit=None, orientation='vertical', loc='left'):

    #Create figure and coclorbar object
    p = figure(name='fig')
    c_b = ColorBar(color_mapper=c_map,
                   major_label_text_font_size='20px',
                   major_label_text_font_style='bold',
                   orientation=orientation,
                   name='cbar')

    #Add units label
    if unit is not None:
        c_b.title = unit
        c_b.title_text_font_size = '18px'
        c_b.title_text_font_style = 'bold'  

    #Display options
    p.add_layout(c_b, loc)
    p.outline_line_color= None
    p.toolbar_location = None

    return p

def mouse_wrapper(ori, sources, images, coords, cross):

    #Determine mask for saving coordinates
    dims = view_info[ori][0]

    #Function to actually update the coordinates
    def mouse_click(event: PointEvent):
        click_coords = [np.int32(event.x), np.int32(event.y)]
        for dim, coord in zip(dims, click_coords):
            coords.data[dim] = [coord]
            cross.data[dim] = [coord] * 2
        update_data(sources, images, coords)

    return mouse_click

#Updates data based on coordinates
def update_data(sources, images, coords):
    for source, image in zip(sources, images):
        for view in views:
            source.data[view] = [np.take(image, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T] 

#Update image data based on thresholding another image
def thresh_wrapper(over_data, over_source, over_ref, thresh_data, coords):
            
   def update_thresh(attrname, old, new):
      over_update = np.copy(over_ref)
      over_update[thresh_data < new] = np.nan
      over_data[:] = np.copy(over_update)
      for view in views:
         over_slice = np.take(over_data, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T
         thresh_slice = np.take(thresh_data, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T
         over_slice[thresh_slice < new] = np.nan
         over_source.data[view] = [over_slice] 

   return update_thresh

#Three rows, two of images and one with plot
def three_row(anat_path, over_paths, thresh_path, meas, unit,
              info_path=None, reg_path=None,
              over_range=[[5, 95]], over_mode=['percentile'],
              anat_range=[2.5, 97.5], anat_mode='percentile',
              over_titles=[None],
              roi_path='./data/winner_wmparc_comb_on_MNI152_2mm_masked.nii.gz',
              over_palettes=['Plasma'],
              over_thresh=[False],
              names_path='./data/wmparc_names.txt'):
              
   #Slider info
   slide_width = 400
   slide_css = InlineStyleSheet(css=".bk-slider-title { font-size: 30px; }")
   slide_wid = Slider(title="-log10(p)-threshold", value=1.3, start=0, end=5, step=0.1, 
                      width=slide_width, stylesheets=[slide_css], name='slide')

   def return_doc(doc):

      def roi_wrapper(sources, images, info, coords, names, fig): 
        
         def update_roi(event: PointEvent):
         
            #Figure out index of roi
            roi_val = np.int32(images[0][np.int32(coords.data['x'][0]),
                                             np.int32(coords.data['y'][0]),
                                             np.int32(coords.data['z'][0])])
            roi_idx = roi_val - 1

            #Update name on scatter plot
            roi_name = names[roi_idx]
            fig.title.text = roi_name.replace(".", " ")

            #Update scatter plot points
            sources[0].data['roi'] = images[1][:, roi_idx]

            #Update scatter plot lines
            roi = [info[roi_name][subj == info['subj']].tolist() for subj in info['subj'].unique()]
            sources[1].data['roi'] = roi

         return update_roi
            
      #Load in anatomical image
      anat_hdr = nib.load(anat_path)
      anat_data = anat_hdr.get_fdata().squeeze()
      anat_x, anat_y, anat_z = np.array(anat_data.shape[0:3])
      anat_dims = {'x':anat_x, 'y':anat_y, 'z':anat_x}
        
      #Load in overlay data
      n_over = len(over_paths)
      over_data = []
      for path in over_paths:
         hdr = nib.load(path)
         data = hdr.get_fdata().squeeze()
         data[data == 0.0] = np.nan
         over_data.append(np.copy(data))

      #Load in thresholding image
      thresh_hdr = nib.load(thresh_path)
      thresh_data = thresh_hdr.get_fdata().squeeze()

      #IO specific to including scatter plot
      if info_path is not None and reg_path is not None:
            
         #Load roi data
         roi_hdr = nib.load(roi_path)
         roi_data = roi_hdr.get_fdata().squeeze()

         #Load in regional data
         reg_hdr = nib.load(reg_path)
         reg_data = reg_hdr.get_fdata().squeeze().T

         #Load in names for each regiion
         roi_names = np.loadtxt(names_path, dtype=np.str_)

         #Read in subject info data frame
         df1 = pd.read_csv(info_path, names=['subj', 'visit', 'cond'])
         df1 = df1.replace(to_replace=['basal', 'hypergly'], value=['Eugly.', 'Hyper.'])
      
         #Construct data frame containing regional data
         df2 = pd.DataFrame(reg_data, columns=roi_names)
         df = pd.concat([df1, df2], axis=1)
         df = df.groupby(['subj', 'cond']).mean().reset_index()

         #Make column sources for spageti plot
         point_source =  ColumnDataSource(data={'cond':df1['cond'],
                                                'subj':df1['subj'],
                                                'roi':reg_data[:, 47]})
         cond = [df['cond'][subj == df['subj']].tolist() for subj in df['subj'].unique()]
         roi = [df['Deep.White.Matter'][subj == df['subj']].tolist() for subj in df['subj'].unique()]
         line_source = ColumnDataSource(data={'cond':cond, 'roi':roi})

      #Define column sources for images
      anat_source = ColumnDataSource(data={'sag':[anat_data[61, :, :].T],
                                           'tra':[anat_data[:, :, 53].T],
                                           'cor':[anat_data[:, 56, :].T]})
      over_sources = []
      for data in over_data:                                     
         over_sources.append(ColumnDataSource(data={'sag':[data[61, :, :].T],
                                                    'tra':[data[:, :, 53].T],
                                                    'cor':[data[:, 56, :].T]}))
      cross_source = ColumnDataSource(data={'x':[61, 61],
                                            'y':[56, 56],
                                            'z':[53, 53],
                                            'anat_x':[0, anat_x],
                                            'anat_y':[0, anat_y],
                                            'anat_z':[0, anat_z]})
      coord_source = ColumnDataSource(data={'x':[61], 'y':[56], 'z':[53]})

      #Compute anatomical range
      anat_mask = anat_data.flatten() != 0
      if anat_mode == 'percentile':
         anat_masked = anat_data.flatten()[anat_mask]
         anat_scale = np.percentile(anat_masked, anat_range)
      else:
         anat_scale = anat_range

      #Compute overlay range if necessary
      over_scales = []
      over_masked = []
      for i in range(n_over):
         over_masked.append(over_data[i].flatten()[anat_mask])
         if over_mode[i] == 'percentile':
            over_scales.append(np.percentile(over_masked[i], over_range[i]))
         else:
            over_scales.append(over_range[i])

      #Define colormaps
      gray_map = LinearColorMapper(low=anat_scale[0], high=anat_scale[1], palette='Greys9',
                                   low_color=(0, 0, 0, 0), nan_color=(0, 0, 0, 0))
      over_maps = []
      for i in range(n_over):
         palette_i = interp_palette(all_palettes[over_palettes[i]][11], 255)
         over_maps.append(LinearColorMapper(low=over_scales[i][0],
                                            high=over_scales[i][1], 
                                            palette=palette_i,
                                            nan_color=(0, 0, 0, 0)))
  
      #Create image plot
      over_rows = []
      for i in range(n_over):
         row_list = []
         for view in ['sag', 'tra', 'cor']:
            row_list.append(overlay_plot(anat_source, over_sources, anat_data,
                                         over_data, i, view, [gray_map, over_maps[i]],
                                         coord_source, cross_source, anat_dims,
                                         title=over_titles[i]))
      
         row_list.append(Spacer(width=25, name='spacer'))
         row_list.append(create_colorbar(over_maps[i], unit=unit))
         if i == 0:
            row_width = sum([p.width for p in row(row_list).children])
            p_r_width = np.int32(row_width / 2.75)
         over_rows.append(row(row_list))
            
         if over_thresh[i] is True: 
         
            #Update image values based on slider
            thresh_func = thresh_wrapper(over_data[i],
                                         over_sources[i],
                                         np.copy(over_data[i]),
                                         np.abs(thresh_data),
                                         coord_source)
            slide_wid.on_change('value', thresh_func)
            thresh_func('init', 1.3, 1.3)
            slide_spac = Spacer(width=p_r_width - int(slide_width / 2), name='spacer')
            slide_row = row(slide_spac, slide_wid)
            over_rows.append(row(Spacer(height=25, name='spacer')))
            over_rows.append(slide_row)

      #Make roi spagetti plot
      if info_path is not None and reg_path is not None:
         p_r = figure(x_range=df1['cond'].unique(),
                      width=p_r_width,
                      height=800,
                      y_axis_label=f'{meas} ({unit})',
                      y_range=[np.min(reg_data) * 0.5, np.max(reg_data) * 1.1])
         p_r.scatter(x='cond', y='roi', source=point_source, size=15)
         p_r.multi_line(xs='cond', ys='roi', source=line_source, line_width=5)
         p_r.axis.major_label_text_font_style = 'normal'
         p_r.axis.major_label_text_font_size  = '32px'
         p_r.yaxis.axis_label_text_font_size = '24px'
         p_r.yaxis.axis_label_text_font_style = 'bold'
         p_r.title = 'Deep White Matter'
         p_r.title.text_font_size = '42px'
         p_r.title.align = 'center'
         p_r.title.text_font_style = 'bold'

         #KDE estimate
         den = img_kde(over_masked[0], over_masked[1])
         dens_palette = interp_palette(all_palettes['Plasma'][11], 255)
         dens_map = LinearColorMapper(low=np.percentile(den, 2.5),
                                      high=np.percentile(den, 97.5),
                                      palette=dens_palette)
  
         #Make column source for scatter plot
         scatter_source = ColumnDataSource(data={'baseline':over_masked[0], 
                                                 'delta':over_masked[1],
                                                 'den':den})
  
         #Make baseline vs. change scatter plot
         p_s = figure(width=p_r_width, height=800,
                      x_axis_label=f'Eugly. {meas} ({unit})',
                      y_axis_label=f'Delta. {meas} ({unit})')     
         p_s.scatter(x='baseline', y='delta', source=scatter_source,
                     color={'field': 'den', 'transform': dens_map}) 
         p_s.axis.major_label_text_font_style = 'normal'
         p_s.axis.major_label_text_font_size  = '32px'
         p_s.axis.axis_label_text_font_size = '24px'
         p_s.axis.axis_label_text_font_style = 'bold'

         #Add roi update event each each plot
         roi_func = roi_wrapper([point_source, line_source], [roi_data, reg_data], df, coord_source, roi_names, p_r)
         for over_row in over_rows:
            for child in over_row.children:
               if child.name == 'image':
                  child.on_event('tap', roi_func)
            
         #Join plots to form final row
         bot_row = row(p_r, Spacer(width=np.int32(row_width / 15), name='spacer'), p_s)
         over_rows.append(Spacer(height=50, name='spacer'))
         over_rows.append(bot_row)

      #Join up all the figures
      out = column(over_rows)

      #Add to document
      doc.add_root(out)

   return return_doc
    
