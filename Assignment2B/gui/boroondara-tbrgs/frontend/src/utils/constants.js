export const SCATS_SITES = [
  { id: '2000', name: 'Warrigal Rd / Toorak Rd',     lat: -37.8456, lng: 145.0856 },
  { id: '2001', name: 'Warrigal Rd / High St',        lat: -37.8390, lng: 145.0856 },
  { id: '2002', name: 'Warrigal Rd / Camberwell Rd',  lat: -37.8320, lng: 145.0856 },
  { id: '2003', name: 'Warrigal Rd / Riversdale Rd',  lat: -37.8250, lng: 145.0856 },
  { id: '3001', name: 'Burke Rd / Toorak Rd',         lat: -37.8456, lng: 145.0723 },
  { id: '3002', name: 'Denmark St / Barkers Rd',      lat: -37.8200, lng: 145.0580 },
  { id: '3003', name: 'Burke Rd / High St',           lat: -37.8390, lng: 145.0723 },
  { id: '3004', name: 'Burke Rd / Camberwell Rd',     lat: -37.8320, lng: 145.0723 },
  { id: '3005', name: 'Burke Rd / Riversdale Rd',     lat: -37.8250, lng: 145.0723 },
  { id: '4001', name: 'Whitehorse Rd / Balwyn Rd',    lat: -37.8180, lng: 145.0830 },
  { id: '4002', name: 'Whitehorse Rd / Burke Rd',     lat: -37.8180, lng: 145.0723 },
  { id: '4003', name: 'Doncaster Rd / Balwyn Rd',     lat: -37.8100, lng: 145.0830 },
  { id: '4004', name: 'Doncaster Rd / Burke Rd',      lat: -37.8100, lng: 145.0723 },
  { id: '5001', name: 'Canterbury Rd / Balwyn Rd',    lat: -37.8260, lng: 145.0830 },
  { id: '5002', name: 'Toorak Rd / Glenferrie Rd',   lat: -37.8456, lng: 145.0612 },
];

export const ML_MODELS = [
  { id: 'lstm',        label: 'LSTM',        fullName: 'Long Short-Term Memory',  color: '#1D9E75', rmse: 12.4, mae: 9.1,  mape: 8.2, r2: 0.91 },
  { id: 'gru',         label: 'GRU',         fullName: 'Gated Recurrent Unit',    color: '#378ADD', rmse: 13.1, mae: 9.8,  mape: 8.9, r2: 0.89 },
  { id: 'transformer', label: 'Transformer', fullName: 'Attention-based Model',   color: '#D85A30', rmse: 11.8, mae: 8.9,  mape: 7.6, r2: 0.93 },
];

export const ROUTE_COLORS = ['#1D9E75', '#378ADD', '#D85A30', '#EF9F27', '#7F77DD'];

export const MAP_CENTER  = { lat: -37.8300, lng: 145.0750 };
export const MAP_ZOOM    = 13;
export const OSM_TILE_URL =
  import.meta.env.VITE_OSM_TILE_URL || 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
export const OSM_ATTRIBUTION =
  import.meta.env.VITE_OSM_ATTRIBUTION || '&copy; OpenStreetMap contributors';
export const OSM_API_KEY = import.meta.env.VITE_OSM_API_KEY || '';

export const DEFAULTS = {
  speedLimit:        60,
  intersectionDelay: 30,
  topK:              5,
  model:             'lstm',
};
