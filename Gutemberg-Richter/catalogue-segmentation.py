import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# rockhound its a package of python 
# to extract Slab2 data(geometry of slab, lat,lon,depth,uncertities-depth,strike,dip)
# you can install with
# conda install conda-forge::rockhound 
# or 
# pip install rockhound
import rockhound as rh
import os
route = os.getcwd()
os.environ["CARTOPY_USER_BACKGROUNDS"] = '/home/alex/BayeChilean_Maximum_Magnitude/'

def isc_read(filename,segmento):
    """
    Read a catalogue from a file and return a pandas DataFrame.
    """
    data=pd.read_csv('/home/alex/BayeChilean_Maximum_Magnitude/Gutemberg-Richter/catalogues/isc-gem-cat.csv', sep=',',date_parser=lambda x: pd.to_datetime(x, format=' %Y-%m-%d %H:%M:%S.%f '),parse_dates=['#         date          '],skiprows=109)
    new_columns = ['date', 'lat', 'lon', 'smajax', 'sminax', 'strike', 'q', 'depth',
    'unc', 'q', 'mw', 'unc', 'q', 's', 'mo', 'fac', 'mo_auth', 'mpp',
    'mpr', 'mrr', 'mrt', 'mtp', 'mtt', 'str1', 'dip1', 'rake1',
    'str2', 'dip2', 'rake2', 'type', 'eventid']
    data.columns = new_columns
    dict_segment= {'IQUIQUE': {'Start': -18.8, 'End': -23.2},
                'ANTOFAGASTA': {'Start': -22.8, 'End': -25.0},
                'VALLENAR': {'Start': -25.8, 'End': -30.2},
                'VALPARAISO': {'Start': -30.0, 'End': -36.0},
                'CONCEPCION': {'Start': -34.0, 'End': -39.0},
                'VALDIVIA': {'Start': -37.5, 'End': -46.0}}
    if segmento in dict_segment.keys():
        data = data[(data['lat'] <= dict_segment[segmento]['Start'] ) & (data['lat'] >= dict_segment[segmento]['End'])]
        data = data[(data['lon'] >= -74.0) & (data['lon'] <= -67.5)]
        data = data[(data['depth'] <= 60.0)]
    else:
        print('Segmento no encontrado')
    return data
class CATALOGOS:
    NOMBRES= {"ISC":' Catálogo de International Seismological Centre', "CSN":' Catálogo de Centro Sismológico Nacional'}  # Definir las opciones disponibles
    SEGMENTO= ['IQUIQUE','ANTOFAGASTA','VALLENAR','VALPARAISO','CONCEPCION','VALDIVIA']
    @staticmethod
    def read_catalogue(nombre,segmento):
        import os
        if nombre == "ISC":
            if segmento in CATALOGOS.SEGMENTO:
                filename=os.getcwd()+'/catalogue/isc-gem-cat.csv'
                return isc_read(filename,segmento)
        elif nombre == "CSN":
            return "Catálogo no disponible, por favor ingrese CATALOGOS.NOMBRES para ver las opciones disponibles"

# Ejemplo de uso
# print(CATALOGOS.NOMBRES)  # Mostrar las opciones disponibles
# print(CATALOGOS.SEGMENTO)  # Mostrar los segmentos disponibles
seismicity={}
min_date={};max_date={}
max_mag={}
for segmento in CATALOGOS.SEGMENTO:
    seismicity[segmento]=CATALOGOS.read_catalogue("ISC",segmento)
    min_date[segmento]=seismicity[segmento]['date'].min()
    max_date[segmento]=seismicity[segmento]['date'].max()
    max_mag[segmento]=seismicity[segmento]['mw'].max()    
absolute_min_date=min(min_date.values())
absolute_max_date=max(max_date.values())
absolute_max_mag=max(max_mag.values())

# Configurar el colormap
cmap = cm.get_cmap('viridis')

# Crear una figura y un eje con proyección PlateCarree
fig = plt.figure(figsize=(5, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Configurar la extensión del mapa para Chile
ax.set_extent([-75, -66, -46, -17], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.STATES.with_scale('10m'))
ax.background_img(name='NaturalEarthRelief', resolution='high')
# Añadir líneas de cuadrícula y título
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.ylabels_right = False
gl.xlabels_top = False
# Añadir un inset con proyección Orthographic
axins = inset_axes(ax, width="45%", height="45%", loc="upper left", 
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=cartopy.crs.Orthographic(central_longitude=-70, central_latitude=-30)))
axins.add_feature(cartopy.feature.COASTLINE)
axins.stock_img()

# Preparar datos para colormap
all_depths = np.concatenate([seismicity[segmento]['depth'].values for segmento in CATALOGOS.SEGMENTO])
norm = mcolors.Normalize(vmin=all_depths.min(), vmax=all_depths.max())

# Colores y tamaños de los puntos
for idx, segmento in enumerate(CATALOGOS.SEGMENTO):
    data = seismicity[segmento]
    lon = data['lon']
    lat = data['lat']
    mag = data['mw']
    depth = data['depth']
    
    # Normalizar y mapear colores según la magnitud
    colors = cmap(norm(depth))
    sizes = 1.2*2**mag # Multiplicar por un factor para ajustar el tamaño
    # Graficar los puntos con colores y tamaños
    ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=colors, s=sizes)
figure_title='Subduction seismicity in Chile \nISC Catalogue (Mw>=5.0, 1906-2020)'
ax.set_title(figure_title)

# Añadir barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Depth (km)')

# Añadir leyenda personalizada para los tamaños de eventos
sizes = [1.2*2**5, 1.2*2**6, 1.2*2**7, 1.2*2**8,1.2*2**9]  # Tamaños de eventos para la leyenda
labels = ['Mw 5', 'Mw 6', 'Mw 7', 'Mw 8','Mw 9']  # Etiquetas para la leyenda
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size), label=label, linestyle='') for size, label in zip(sizes, labels)]

# Añadir leyenda en la parte inferior
ax.legend(handles=handles, loc='lower center', ncol=3,title='Simbology',fancybox=True,bbox_to_anchor=(0.5, -0.15))

# Mostrar el gráfico
plt.savefig('seismicity-ISC.png',dpi=300)