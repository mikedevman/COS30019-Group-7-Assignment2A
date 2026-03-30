export const SCATS_SITES = [
  { id: '2000', name: 'Toorak Rd / Warrigal Rd', lat: -37.8516827, lng: 145.0943457 },
  { id: '2200', name: 'Maroondah Hwy / Union Rd', lat: -37.81631, lng: 145.09812 },
  { id: '2820', name: 'Princess St / Chandler Hwy', lat: -37.79477, lng: 145.03077 },
  { id: '2825', name: 'Burke Rd / Eastern Fwy', lat: -37.78661, lng: 145.06202 },
  { id: '2827', name: 'Eastern Fwy Offramp / Bulleen Rd', lat: -37.78093, lng: 145.07733 },
  { id: '2846', name: 'Wills St / High St', lat: -37.8612671, lng: 145.058038 },
  { id: '3001', name: 'Barkers Rd / Church St', lat: -37.81441, lng: 145.02243 },
  { id: '3002', name: 'Barkers Rd / Denmark St', lat: -37.81489, lng: 145.02663 },
  { id: '3120', name: 'Rathmines Rd / Burke Rd', lat: -37.82264, lng: 145.05734 },
  { id: '3122', name: 'Canterbury Rd / Stanhope Gv', lat: -37.82379, lng: 145.06466 },
  { id: '3126', name: 'Canterbury Rd / Warrigal Rd', lat: -37.82778, lng: 145.09885 },
  { id: '3127', name: 'Canterbury Rd / Balwyn Rd', lat: -37.82506, lng: 145.078 },
  { id: '3180', name: 'Doncaster Rd / Balwyn Rd', lat: -37.79611, lng: 145.08372 },
  { id: '3662', name: 'Studley Park Rd / Princess St', lat: -37.80876, lng: 145.02757 },
  { id: '3682', name: 'Riversdale Rd / Warrigal Rd', lat: -37.83695, lng: 145.09699 },
  { id: '3685', name: 'Warrigal Rd / Highbury Rd', lat: -37.85467, lng: 145.09384 },
  { id: '3804', name: 'Riversdale Rd / Trafalgar Rd', lat: -37.83331, lng: 145.06247 },
  { id: '3812', name: 'Camberwell Rd / Trafalgar Rd', lat: -37.83738, lng: 145.06119 },
  { id: '4030', name: 'Kilby Rd / Burke Rd', lat: -37.79561, lng: 145.06251 },
  { id: '4032', name: 'Harp Rd / Burke Rd', lat: -37.80202, lng: 145.06127 },
  { id: '4034', name: 'Cotham Rd / Burke Rd', lat: -37.81147, lng: 145.05946 },
  { id: '4035', name: 'Barkers Rd / Burke Rd', lat: -37.8172654, lng: 145.0583603 },
  { id: '4040', name: 'Camberwell Rd / Burke Rd', lat: -37.83256, lng: 145.05545 },
  { id: '4043', name: 'Toorak Rd / Burke Rd', lat: -37.84683, lng: 145.05275 },
  { id: '4051', name: 'Doncaster Rd / Severn St', lat: -37.79419, lng: 145.0696 },
  { id: '4057', name: 'Belmore Rd / Balwyn Rd', lat: -37.80431, lng: 145.08197 },
  { id: '4063', name: 'Whitehorse Rd / Balwyn Rd', lat: -37.81404, lng: 145.0801 },
  { id: '4262', name: 'Bridge Rd / Burwood Rd', lat: -37.82155, lng: 145.01503 },
  { id: '4263', name: 'Burwood Rd / Power St', lat: -37.8228462, lng: 145.0251292 },
  { id: '4264', name: 'Burwood Rd / Glenferrie Rd', lat: -37.82389, lng: 145.03409 },
  { id: '4266', name: 'Burwood Rd / Auburn Rd', lat: -37.82529, lng: 145.04387 },
  { id: '4270', name: 'Riversdale Rd / Glenferrie Rd', lat: -37.82951, lng: 145.03304 },
  { id: '4272', name: 'Riversdale Rd / Tooronga Rd', lat: -37.83186, lng: 145.04668 },
  { id: '4273', name: 'Toorak Rd / Tooronga Rd', lat: -37.84632, lng: 145.04378 },
  { id: '4321', name: 'Valerie St / High St', lat: -37.800776, lng: 145.0494611 },
  { id: '4324', name: 'Cotham Rd / Glenferrie Rd', lat: -37.809274, lng: 145.037306 },
  { id: '4335', name: 'High St / Charles St', lat: -37.80624, lng: 145.03518 },
  { id: '4812', name: 'Swan St / Madden Gv', lat: -37.82859, lng: 145.01644 },
  { id: '4821', name: 'Victoria St / Burnley St', lat: -37.81285, lng: 145.00849 },
  { id: '970', name: 'High St / Warrigal Rd', lat: -37.86703, lng: 145.09159 },
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
