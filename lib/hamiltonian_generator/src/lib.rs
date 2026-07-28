#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![feature(int_roundings)]
use std::{collections::HashMap, f64::consts::PI};

use num_complex::{Complex, Complex64};
use pyo3::prelude::*;

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}
#[must_use]
const fn hermite_coefficient(n: i64, m: i64) -> i64 {
    match (n, m) {
        (0, 0) => 1,
        (1, 0) => 0,
        (1, 1) => 2,
        (_, 0) => -hermite_coefficient(n - 1, m + 1),
        (n, m) if n >= m => {
            2 * hermite_coefficient(n - 1, m - 1) - (m + 1) * hermite_coefficient(n - 1, m + 1)
        }
        _ => 0,
    }
}
#[allow(clippy::cast_precision_loss)]
fn hermite_val(x: f64, n: u32) -> f64 {
    (0..=n)
        .map(|m| (hermite_coefficient(n.into(), m.into()) as f64) * x.powi(m.try_into().unwrap()))
        .sum()
}

fn calculate_sho_wavefunction(z_points: &Vec<f64>, sho_omega: f64, mass: f64, n: u32) -> Vec<f64> {
    let norm = ((sho_omega * mass) / REDUCED_PLANCK_CONSTANT).sqrt();
    let factorial: f64 = factorial(n.into()) as f64;
    let prefactor = (norm / factorial) / (PI.sqrt() * 2_f64.powi(n.try_into().unwrap()));
    let sqrt_prefactor = prefactor.sqrt();

    z_points
        .iter()
        .map(|p| -> f64 {
            let normalized_p = p * norm;
            let hermite_val = hermite_val(normalized_p, n);
            sqrt_prefactor * hermite_val * f64::exp(-normalized_p.powi(2) / 2.0)
        })
        .collect()
}

const PLANCK_CONSTANT: f64 = 6.626_070_15E-34;
const REDUCED_PLANCK_CONSTANT: f64 = PLANCK_CONSTANT / (2.0 * PI);

struct EigenstateResolution(i64, i64, usize);

impl EigenstateResolution {
    fn x0_coordinates(&self) -> impl Iterator<Item = i64> {
        let max_x0 = self.0.div_ceil(2);
        let min_x0 = self.0.div_floor(2);
        (0..max_x0).chain(-min_x0..0)
    }
    fn x1_coordinates(&self) -> impl Iterator<Item = i64> {
        let max_x1 = self.1.div_ceil(2);
        let min_x1 = self.1.div_floor(2);
        (0..max_x1).chain(-min_x1..0)
    }
    fn coordinates(&self) -> Vec<(i64, i64, usize)> {
        self.x0_coordinates()
            .flat_map(|x| {
                self.x1_coordinates()
                    .flat_map(move |y| (0..self.2).map(move |z| (x, y, z)))
            })
            .collect()
    }
}

struct EigenstateConfig {
    sho_omega: f64,
    mass: f64,
    delta_x0: (f64, f64),
    delta_x1: (f64, f64),
}

struct Eigenstate {
    config: EigenstateConfig,
    resolution: EigenstateResolution,
    vector: Vec<num_complex::Complex64>,
    kx: f64,
    ky: f64,
}

impl Eigenstate {
    fn calculate_wavefunction(&self, points: &Vec<[f64; 3]>) -> Vec<Complex64> {
        let coordinates = self.resolution.coordinates();
        let z_points: Vec<f64> = points.iter().map(|p| p[2]).collect();

        let cache: Vec<Vec<f64>> = (0..self.resolution.2)
            .map(|nz| -> Vec<f64> {
                calculate_sho_wavefunction(
                    &z_points,
                    self.config.sho_omega,
                    self.config.mass,
                    nz as u32,
                )
            })
            .collect();

        let dkx0 = self.config.dkx0();
        let dkx1 = self.config.dkx1();

        let mut out: Vec<Complex64> = vec![Complex::default(); points.len()];
        for (eig, (nkx0, nkx1, nz)) in self.vector.iter().zip(coordinates) {
            for (i, wfn) in cache[nz].iter().enumerate() {
                out[i] += wfn
                    * eig
                    * Complex {
                        re: 0.0,
                        im: (((nkx0 as f64) * dkx0.0) + ((nkx1 as f64) * dkx1.0) + self.kx)
                            * points[i][0]
                            + (((nkx0 as f64) * dkx0.1) + ((nkx1 as f64) * dkx1.1) + self.ky)
                                * points[i][1],
                    }
                    .exp();
            }
        }

        out
    }
}

impl EigenstateConfig {
    fn dk_prefactor(&self) -> f64 {
        let x0_part = self.delta_x0.0 * self.delta_x1.1;
        let x1_part = self.delta_x0.1 * self.delta_x1.0;
        (2.0 * PI) / (x0_part - x1_part)
    }
    fn dkx0(&self) -> (f64, f64) {
        let prefactor = self.dk_prefactor();
        (prefactor * self.delta_x1.1, -prefactor * self.delta_x1.0)
    }

    fn dkx1(&self) -> (f64, f64) {
        let prefactor = self.dk_prefactor();
        (-prefactor * self.delta_x0.1, prefactor * self.delta_x0.0)
    }
}

struct SurfaceHamiltonian {
    sho_config: EigenstateConfig,
    resolution: EigenstateResolution,
    ft_potential: Vec<Vec<Vec<Complex64>>>,
    dz: f64,
    z_offset: f64,
}

impl SurfaceHamiltonian {
    fn get_nx0(&self) -> usize {
        self.ft_potential.len()
    }

    fn get_nx1(&self) -> usize {
        self.ft_potential[0].len()
    }

    fn get_nz(&self) -> usize {
        self.ft_potential[0][0].len()
    }

    fn get_z_points(&self) -> Vec<f64> {
        (0..self.get_nz())
            .map(|i| self.dz.mul_add(i as f64, self.z_offset))
            .collect()
    }

    fn calculate_sho_wavefunction(&self, n: u32) -> Vec<f64> {
        calculate_sho_wavefunction(
            &self.get_z_points(),
            self.sho_config.sho_omega,
            self.sho_config.mass,
            n,
        )
    }

    fn calculate_off_diagonal_energies(&self) -> Vec<Vec<Complex64>> {
        let coordinates = self.resolution.coordinates();
        let cache: Vec<Vec<f64>> = (0..self.resolution.2)
            .map(|nz| -> Vec<f64> { self.calculate_sho_wavefunction(nz.try_into().unwrap()) })
            .collect();

        let mut g_points: HashMap<(usize, usize, usize, usize), Complex64> = HashMap::new();

        coordinates
            .iter()
            .map(|(nkx0_1, nkx1_1, nz1)| -> Vec<Complex64> {
                coordinates
                    .iter()
                    .map(|(nkx0_2, nkx1_2, nz2)| -> Complex64 {
                        let n_dkx0 = usize::try_from(
                            (nkx0_2 - nkx0_1).rem_euclid(i64::try_from(self.get_nx0()).unwrap()),
                        )
                        .unwrap();
                        let n_dkx1 = usize::try_from(
                            (nkx1_2 - nkx1_1).rem_euclid(i64::try_from(self.get_nx1()).unwrap()),
                        )
                        .unwrap();
                        if let Some(a) = g_points.get(&(n_dkx0, n_dkx1, *nz1, *nz2)) {
                            return *a;
                        }

                        let ft_pot_points = &self.ft_potential[n_dkx0][n_dkx1];

                        let sho1: &Vec<f64> = &cache[*nz1];
                        let sho2: &Vec<f64> = &cache[*nz2];

                        let out = ft_pot_points
                            .iter()
                            .zip(sho1)
                            .zip(sho2)
                            .map(|((i, j), k)| i * j * k)
                            .sum::<Complex64>()
                            * self.dz;
                        g_points.insert((n_dkx0, n_dkx1, *nz1, *nz2), out);
                        out
                    })
                    .collect()
            })
            .collect()
    }
}

struct SurfaceHamiltonian2 {
    resolution: EigenstateResolution,
    ft_potential: Vec<Vec<Vec<Complex64>>>,
    eigenstates_z: Vec<Vec<Complex64>>,
}

impl SurfaceHamiltonian2 {
    fn get_nx0(&self) -> usize {
        self.ft_potential.len()
    }

    fn get_nx1(&self) -> usize {
        self.ft_potential[0].len()
    }

    fn calculate_off_diagonal_energies(&self) -> Vec<Vec<Complex64>> {
        let coordinates = self.resolution.coordinates();

        let mut g_points: HashMap<(usize, usize, usize, usize), Complex64> = HashMap::new();

        coordinates
            .iter()
            .map(|(nkx0_1, nkx1_1, nz1)| -> Vec<Complex64> {
                coordinates
                    .iter()
                    .map(|(nkx0_2, nkx1_2, nz2)| -> Complex64 {
                        let n_dkx0 = usize::try_from(
                            (nkx0_2 - nkx0_1).rem_euclid(i64::try_from(self.get_nx0()).unwrap()),
                        )
                        .unwrap();
                        let n_dkx1 = usize::try_from(
                            (nkx1_2 - nkx1_1).rem_euclid(i64::try_from(self.get_nx1()).unwrap()),
                        )
                        .unwrap();
                        if let Some(a) = g_points.get(&(n_dkx0, n_dkx1, *nz1, *nz2)) {
                            return *a;
                        }

                        let ft_pot_points = &self.ft_potential[n_dkx0][n_dkx1];

                        let sho1: &Vec<Complex64> = &self.eigenstates_z[*nz1];
                        let sho2: &Vec<Complex64> = &self.eigenstates_z[*nz2];

                        let out = ft_pot_points
                            .iter()
                            .zip(sho1)
                            .zip(sho2)
                            .map(|((i, j), k)| i * j * k.conj())
                            .sum::<Complex64>();
                        g_points.insert((n_dkx0, n_dkx1, *nz1, *nz2), out);
                        out
                    })
                    .collect()
            })
            .collect()
    }
}

#[pyfunction]
fn get_hermite_val(x: f64, n: u32) -> f64 {
    hermite_val(x, n)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn get_sho_wavefunction(z_points: Vec<f64>, sho_omega: f64, mass: f64, n: u32) -> Vec<f64> {
    calculate_sho_wavefunction(&z_points, sho_omega, mass, n)
}

#[pyfunction]
fn calculate_off_diagonal_energies(
    ft_potential: Vec<Vec<Vec<Complex64>>>,
    resolution: [usize; 3],
    dz: f64,
    mass: f64,
    sho_omega: f64,
    z_offset: f64,
) -> Vec<Vec<Complex64>> {
    let sho_config = EigenstateConfig {
        mass,
        sho_omega,
        delta_x0: (1.0, 0.0),
        delta_x1: (0.0, 1.0),
    };
    let hamiltonian = SurfaceHamiltonian {
        dz,
        ft_potential,
        resolution: EigenstateResolution(
            resolution[0].try_into().unwrap(),
            resolution[1].try_into().unwrap(),
            resolution[2],
        ),
        sho_config,
        z_offset,
    };
    hamiltonian.calculate_off_diagonal_energies()
}

#[pyfunction]
fn calculate_off_diagonal_energies2(
    ft_potential: Vec<Vec<Vec<Complex64>>>,
    eigenstates_z: Vec<Vec<Complex64>>,
    resolution: [usize; 3],
) -> Vec<Vec<Complex64>> {
    let hamiltonian = SurfaceHamiltonian2 {
        ft_potential,
        resolution: EigenstateResolution(
            resolution[0].try_into().unwrap(),
            resolution[1].try_into().unwrap(),
            resolution[2],
        ),
        eigenstates_z,
    };
    hamiltonian.calculate_off_diagonal_energies()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn get_eigenstate_wavefunction(
    resolution: [usize; 3],
    delta_x0: (f64, f64),
    delta_x1: (f64, f64),
    mass: f64,
    sho_omega: f64,
    kx: f64,
    ky: f64,
    vector: Vec<Complex64>,
    points: Vec<[f64; 3]>,
) -> Vec<Complex64> {
    let eigenstate = Eigenstate {
        config: EigenstateConfig {
            sho_omega,
            mass,
            delta_x0,
            delta_x1,
        },
        kx,
        ky,
        resolution: EigenstateResolution(
            resolution[0].try_into().unwrap(),
            resolution[1].try_into().unwrap(),
            resolution[2],
        ),
        vector,
    };

    eigenstate.calculate_wavefunction(&points)
}

/// A Python module implemented in Rust.
#[pymodule]
fn hamiltonian_generator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_off_diagonal_energies, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_off_diagonal_energies2, m)?)?;
    m.add_function(wrap_pyfunction!(get_sho_wavefunction, m)?)?;
    m.add_function(wrap_pyfunction!(get_hermite_val, m)?)?;
    m.add_function(wrap_pyfunction!(get_eigenstate_wavefunction, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{EigenstateConfig, EigenstateResolution, SurfaceHamiltonian};

    #[test]
    fn test_calculate_off_diagonal_energies() {
        let sho_config = EigenstateConfig {
            mass: 1.0,
            sho_omega: 1.0,
            delta_x0: (1.0, 0.0),
            delta_x1: (0.0, 1.0),
        };
        let hamiltonian = SurfaceHamiltonian {
            dz: 1.0,
            ft_potential: vec![
                vec![vec![0.0.into(), 0.0.into()], vec![0.0.into(), 0.0.into()]],
                vec![vec![0.0.into(), 0.0.into()], vec![0.0.into(), 0.0.into()]],
            ],
            resolution: EigenstateResolution(2, 2, 2),
            sho_config,
            z_offset: 0.0,
        };

        hamiltonian.calculate_off_diagonal_energies();
    }

    #[test]
    fn test_get_resolution_coordinates() {
        let resolution = EigenstateResolution(5, 4, 2);
        assert!(resolution
            .x0_coordinates()
            .zip(vec![0, 1, 2, -2, -1])
            .all(|(l, r)| l == r));

        assert!(resolution
            .x1_coordinates()
            .zip(vec![0, 1, -2, -1])
            .all(|(l, r)| l == r));
    }
}
