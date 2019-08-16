use std::fs::File;
use std::io::BufWriter;
use std::io::Write as IoWrite;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, UnitDisc, UnitSphere};
use rand_pcg;

#[derive(Copy, Clone, Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn point_at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Copy, Clone, Debug)]
struct Hit {
    point: Vec3,
    normal: Vec3,
    material: Material,
}

trait Object {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit>;
}

#[derive(Copy, Clone, Debug)]
struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
        let oc = ray.origin - self.center;
        let b = oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - c;
        if discriminant > 0.0 {
            let sqrt = discriminant.sqrt();
            let t = -b - sqrt;
            if t < t_max && t > t_min {
                return Some(t);
            }

            let t = -b + sqrt;
            if t < t_max && t > t_min {
                return Some(t);
            }
        }
        None
    }
}

struct World {
    // "Optimizing" xd, there are only spheres here! Only the world generation changes.
    // objects: Vec<Box<dyn Object>>
    objects: Vec<Sphere>,
    materials: Vec<Material>,
}

impl World {
    fn random_scene<R: Rng + ?Sized>(rng: &mut R) -> World {
        let mut objects = Vec::new();
        let mut materials = Vec::new();

        objects.push(Sphere {
            center: Vec3::new(0.0, -1000.0, 0.0),
            radius: 1000.0,
        });
        materials.push(Material::Lambert {
            albedo: Vec3::new(0.5, 0.5, 0.5),
        });

        for a in -11..11 {
            for b in -11..11 {
                let center = Vec3::new(
                    a as f32 + rng.gen::<f32>() * 0.9,
                    0.2,
                    b as f32 + 0.9 * rng.gen::<f32>(),
                );

                if (center - Vec3::new(4.0, 0.2, 0.0)).mag() > 0.9 {
                    let choose_mat = rng.gen::<f32>();
                    materials.push(if choose_mat < 0.8 {
                        Material::Lambert {
                            albedo: Vec3::new(
                                rng.gen::<f32>() * rng.gen::<f32>(),
                                rng.gen::<f32>() * rng.gen::<f32>(),
                                rng.gen::<f32>() * rng.gen::<f32>(),
                            ),
                        }
                    } else if choose_mat < 0.95 {
                        Material::Metal {
                            albedo: Vec3::new(
                                0.5 * (1.0 + rng.gen::<f32>()),
                                0.5 * (1.0 + rng.gen::<f32>()),
                                0.5 * (1.0 + rng.gen::<f32>()),
                            ),
                            roughness: 0.5 * rng.gen::<f32>(),
                        }
                    } else {
                        Material::Dielectric {
                            refractive_index: 1.5,
                        }
                    });

                    objects.push(Sphere {
                        center: center,
                        radius: 0.2,
                    });
                }
            }
        }

        objects.push(Sphere {
            center: Vec3::new(0.0, 1.0, 0.0),
            radius: 1.0,
        });
        materials.push(Material::Dielectric {
            refractive_index: 1.5,
        });

        objects.push(Sphere {
            center: Vec3::new(-4.0, 1.0, 0.0),
            radius: 1.0,
        });
        materials.push(Material::Lambert {
            albedo: Vec3::new(0.4, 0.2, 0.1),
        });

        objects.push(Sphere {
            center: Vec3::new(4.0, 1.0, 0.0),
            radius: 1.0,
        });
        materials.push(Material::Metal {
            albedo: Vec3::new(0.7, 0.6, 0.5),
            roughness: 0.0,
        });

        World {
            objects: objects,
            materials: materials,
        }
    }
}

impl Object for World {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let mut nearest_t: f32 = t_max;
        let mut hit_index: Option<usize> = None;
        for (i, object) in self.objects.iter().enumerate() {
            if let Some(t) = object.hit(ray, t_min, nearest_t) {
                nearest_t = t;
                hit_index = Some(i);
            }
        }

        if let Some(index) = hit_index {
            let sphere = self.objects[index];
            let material = self.materials[index];
            let p = ray.point_at(nearest_t);
            let n = (p - sphere.center) / sphere.radius;
            return Some(Hit {
                point: p,
                normal: n,
                material: material,
            });
        }

        None
    }
}

struct Camera {
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    fn new(
        look_from: Vec3,
        look_at: Vec3,
        up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_distance: f32,
    ) -> Self {
        let theta = vertical_fov * std::f32::consts::PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;

        let w = (look_from - look_at).unit();
        let u = up.cross(w).unit();
        let v = w.cross(u);

        Camera {
            lens_radius: aperture / 2.0,
            lower_left: look_from
                - half_width * focus_distance * u
                - half_height * focus_distance * v
                - focus_distance * w,
            horizontal: 2.0 * half_width * focus_distance * u,
            vertical: 2.0 * half_height * focus_distance * v,
            origin: look_from,
            u: u,
            v: v,
        }
    }

    fn generate_ray<R: Rng + ?Sized>(&self, s: f32, t: f32, rng: &mut R) -> Ray {
        let rd: [f32; 2] = UnitDisc.sample(rng);
        let offset: Vec3 = self.u * rd[0] * self.lens_radius + self.v * rd[1] * self.lens_radius;
        Ray {
            origin: self.origin + offset,
            direction: (self.lower_left + self.horizontal * s + self.vertical * t
                - self.origin
                - offset)
                .unit(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Material {
    Lambert { albedo: Vec3 },
    Metal { albedo: Vec3, roughness: f32 },
    Dielectric { refractive_index: f32 },
}

fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = v.unit();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if discriminant > 0.0 {
        return Some(ni_over_nt * (uv - n * dt) - n * discriminant.sqrt());
    }
    None
}

fn schlick(cosine: f32, refractive_index: f32) -> f32 {
    let r0 = (1.0 - refractive_index) / (1.0 + refractive_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

fn scatter<R: Rng + ?Sized>(ray: &Ray, hit: &Hit, rng: &mut R) -> Option<(Vec3, Ray)> {
    match hit.material {
        Material::Lambert { albedo } => {
            let target = hit.point + hit.normal + Vec3::from(UnitSphere.sample(rng));
            let scattered = Ray {
                origin: hit.point,
                direction: (target - hit.point).unit(),
            };
            return Some((albedo, scattered));
        }
        Material::Metal { albedo, roughness } => {
            let reflected = ray.direction.unit().reflect(hit.normal)
                + roughness * Vec3::from(UnitSphere.sample(rng));
            if reflected.dot(hit.normal) > 0.0 {
                let scattered = Ray {
                    origin: hit.point,
                    direction: reflected.unit(),
                };
                return Some((albedo, scattered));
            }
        }
        Material::Dielectric { refractive_index } => {
            let (outward_normal, ni_over_nt, cosine) = if ray.direction.dot(hit.normal) > 0.0 {
                let cos = refractive_index * ray.direction.dot(hit.normal) / ray.direction.mag();
                (-hit.normal, refractive_index, cos)
            } else {
                let cos = -(ray.direction.dot(hit.normal)) / ray.direction.mag();
                (hit.normal, 1.0 / refractive_index, cos)
            };

            if let Some(refracted) = refract(ray.direction, outward_normal, ni_over_nt) {
                let reflect_probability = schlick(cosine, refractive_index);
                if rng.gen::<f32>() >= reflect_probability {
                    return Some((
                        Vec3::new(1.0, 1.0, 1.0),
                        Ray {
                            origin: hit.point,
                            direction: refracted.unit(),
                        },
                    ));
                }
            }

            let reflected = ray.direction.reflect(hit.normal);
            return Some((
                Vec3::new(1.0, 1.0, 1.0),
                Ray {
                    origin: hit.point,
                    direction: reflected,
                },
            ));
        }
    }

    None
}

fn trace<R: Rng + ?Sized>(ray: Ray, world: &World, rng: &mut R, depth: i32) -> Vec3 {
    if let Some(hit) = world.hit(&ray, 0.001, std::f32::MAX) {
        if depth < 50 {
            if let Some((attenuation, scattered)) = scatter(&ray, &hit, rng) {
                return attenuation * trace(scattered, world, rng, depth + 1);
            }
        }
        Vec3::new(0.0, 0.0, 0.0)
    } else {
        let unit_direction = ray.direction.unit();
        let t = 0.5 * (unit_direction.y() + 1.0);
        Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[Vec3]) -> std::io::Result<()> {
    let file = File::create(path)?;

    let mut writer = BufWriter::new(file);

    write!(writer, "P3\n{} {}\n255\n", width, height)?;

    for y in (0..height).rev() {
        for x in 0..width {
            let pixel = pixels[y * width + x];
            let r = (pixel.r().sqrt() * 255.99) as i32;
            let g = (pixel.g().sqrt() * 255.99) as i32;
            let b = (pixel.b().sqrt() * 255.99) as i32;
            write!(writer, "{} {} {}\n", r, g, b)?;
        }
    }

    Ok(())
}

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const SAMPLES: usize = 128;

fn main() {
    let mut pixels: [Vec3; WIDTH * HEIGHT] = [Vec3::default(); WIDTH * HEIGHT];

    let mut rng = rand_pcg::Pcg32::seed_from_u64(1337);

    let from = Vec3::new(8.0, 3.0, 2.0);
    let at = Vec3::new(0.0, 0.0, 0.0);
    let camera = Camera::new(
        from,
        at,
        Vec3::new(0.0, 1.0, 0.0),
        45.0,
        WIDTH as f32 / HEIGHT as f32,
        0.05,
        (from - at).mag(),
    );

    let world = World::random_scene(&mut rng);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut colour = Vec3::default();
            for _s in 0..SAMPLES {
                let u = (x as f32 + rng.gen::<f32>()) / WIDTH as f32;
                let v = (y as f32 + rng.gen::<f32>()) / HEIGHT as f32;
                let ray = camera.generate_ray(u, v, &mut rng);
                colour += trace(ray, &world, &mut rng, 0);
            }
            colour /= SAMPLES as f32;
            pixels[y * WIDTH + x] = colour;
        }
    }

    let path = Path::new("output.ppm");
    write_ppm(path, WIDTH, HEIGHT, &pixels).unwrap();
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x: x, y: y, z: z }
    }

    pub fn unit(self) -> Vec3 {
        self * (1.0 / self.mag())
    }

    pub fn mag(self) -> f32 {
        self.mag_sq().sqrt()
    }

    pub fn mag_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            -(self.x * rhs.z - self.z * rhs.x),
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn reflect(self, normal: Self) -> Self {
        self - 2.0 * self.dot(normal) * normal
    }

    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }

    pub fn z(&self) -> f32 {
        self.z
    }

    pub fn r(&self) -> f32 {
        self.x
    }

    pub fn g(&self) -> f32 {
        self.y
    }

    pub fn b(&self) -> f32 {
        self.z
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(values: [f32; 3]) -> Self {
        Vec3::new(values[0], values[1], values[2])
    }
}

impl From<(f32, f32, f32)> for Vec3 {
    fn from(values: (f32, f32, f32)) -> Self {
        Vec3::new(values.0, values.1, values.2)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Vec3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div for Vec3 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Vec3::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        let rcp = 1.0 / rhs;
        Vec3::new(self.x * rcp, self.y * rcp, self.z * rcp)
    }
}

impl DivAssign for Vec3 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }
}
