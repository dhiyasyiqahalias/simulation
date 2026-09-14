/**
 * tourism-data.js - Sustainable Tourism 2023 Dataset for Malaysian States (DOSM Datathon)
 */

export const MalaysianTourismData = [
  {
    state: 'W.P. Putrajaya',
    visitors: 1900.379, // in thousands
    receipts: 669.729,   // in RM million
    stay: 2.0797,        // average length of stay in days
    population: 118.8,   // in thousands
    density: 15.996,     // visitors per resident
    spend: 352.42,       // spend per visitor in RM
    tier: 'Overcrowded'  // DOSM rule-based classification
  },
  {
    state: 'Melaka',
    visitors: 15558.661,
    receipts: 6438.411,
    stay: 2.2340,
    population: 1028.3,
    density: 15.130,
    spend: 413.82,
    tier: 'Overcrowded'
  },
  {
    state: 'Negeri Sembilan',
    visitors: 14959.470,
    receipts: 4777.281,
    stay: 2.4184,
    population: 1224.9,
    density: 12.213,
    spend: 319.35,
    tier: 'Overcrowded'
  },
  {
    state: 'W.P. Kuala Lumpur',
    visitors: 22232.643,
    receipts: 10995.324,
    stay: 2.6963,
    population: 2005.7,
    density: 11.085,
    spend: 494.56,
    tier: 'Balanced'
  },
  {
    state: 'Pahang',
    visitors: 16455.567,
    receipts: 6733.344,
    stay: 2.1272,
    population: 1643.2,
    density: 10.014,
    spend: 409.18,
    tier: 'Balanced'
  },
  {
    state: 'Terengganu',
    visitors: 11760.516,
    receipts: 4192.317,
    stay: 2.5941,
    population: 1210.0,
    density: 9.719,
    spend: 356.47,
    tier: 'Balanced'
  },
  {
    state: 'Pulau Pinang',
    visitors: 13127.587,
    receipts: 5814.735,
    stay: 2.4003,
    population: 1772.6,
    density: 7.406,
    spend: 442.94,
    tier: 'Balanced'
  },
  {
    state: 'Sarawak',
    visitors: 17901.190,
    receipts: 6942.673,
    stay: 3.4815,
    population: 2502.3,
    density: 7.154,
    spend: 387.83,
    tier: 'Balanced'
  },
  {
    state: 'Perak',
    visitors: 17107.518,
    receipts: 5641.024,
    stay: 2.3458,
    population: 2541.2,
    density: 6.732,
    spend: 329.74,
    tier: 'Balanced'
  },
  {
    state: 'Perlis',
    visitors: 1950.771,
    receipts: 583.870,
    stay: 2.2064,
    population: 293.1,
    density: 6.656,
    spend: 299.30,
    tier: 'Balanced'
  },
  {
    state: 'Kedah',
    visitors: 13444.055,
    receipts: 4206.619,
    stay: 2.0272,
    population: 2189.3,
    density: 6.141,
    spend: 312.90,
    tier: 'Balanced'
  },
  {
    state: 'Sabah',
    visitors: 16080.074,
    receipts: 6712.515,
    stay: 2.6714,
    population: 3596.7,
    density: 4.471,
    spend: 417.44,
    tier: 'Balanced'
  },
  {
    state: 'Kelantan',
    visitors: 7549.370,
    receipts: 3228.385,
    stay: 3.0735,
    population: 1859.8,
    density: 4.059,
    spend: 427.64,
    tier: 'Balanced'
  },
  {
    state: 'Johor',
    visitors: 15804.890,
    receipts: 6699.119,
    stay: 2.1761,
    population: 4107.2,
    density: 3.848,
    spend: 423.86,
    tier: 'Under-visited'
  },
  {
    state: 'Selangor',
    visitors: 27579.478,
    receipts: 11102.713,
    stay: 2.2421,
    population: 7209.7,
    density: 3.825,
    spend: 402.57,
    tier: 'Under-visited'
  },
  {
    state: 'W.P. Labuan',
    visitors: 331.360,
    receipts: 194.064,
    stay: 2.8863,
    population: 99.0,
    density: 3.347,
    spend: 585.66,
    tier: 'Under-visited'
  }
];

export const TourismDimensions = {
  density: {
    label: 'Density (visitors/resident)',
    unit: 'visitors/resident',
    format: (v) => `${v.toFixed(2)}`
  },
  spend: {
    label: 'Spend per Visitor',
    unit: 'RM',
    format: (v) => `RM ${v.toFixed(0)}`
  },
  visitors: {
    label: "Domestic Visitors ('000)",
    unit: 'k visitors',
    format: (v) => `${v.toLocaleString()}k`
  },
  receipts: {
    label: 'Total Receipts (RM mil)',
    unit: 'RM mil',
    format: (v) => `RM ${v.toLocaleString()}M`
  },
  stay: {
    label: 'Avg Length of Stay',
    unit: 'days',
    format: (v) => `${v.toFixed(2)} days`
  },
  population: {
    label: "Population ('000)",
    unit: 'k people',
    format: (v) => `${v.toLocaleString()}k`
  }
};

/**
 * Project Tourism Data to 2D Canvas coordinate space
 */
export function projectTourismPoints(xDim = 'density', yDim = 'spend', width = 800, height = 600, padding = 60) {
  const xVals = MalaysianTourismData.map(d => d[xDim]);
  const yVals = MalaysianTourismData.map(d => d[yDim]);

  const minX = Math.min(...xVals);
  const maxX = Math.max(...xVals);
  const minY = Math.min(...yVals);
  const maxY = Math.max(...yVals);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const usableW = width - padding * 2;
  const usableH = height - padding * 2;

  return MalaysianTourismData.map((d, idx) => {
    // Invert Y so higher values appear at the top
    const normX = (d[xDim] - minX) / rangeX;
    const normY = (d[yDim] - minY) / rangeY;

    const x = padding + normX * usableW;
    const y = height - padding - normY * usableH;

    return {
      x,
      y,
      trueCluster: d.tier === 'Overcrowded' ? 0 : d.tier === 'Balanced' ? 1 : 2,
      meta: {
        state: d.state,
        tier: d.tier,
        density: d.density,
        spend: d.spend,
        visitors: d.visitors,
        receipts: d.receipts,
        stay: d.stay,
        population: d.population,
        rawX: d[xDim],
        rawY: d[yDim],
        xDim,
        yDim
      }
    };
  });
}
