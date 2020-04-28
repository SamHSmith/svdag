use vulkano::buffer::CpuBufferPool;
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::AttachmentImage;
use vulkano::image::StorageImage;
use vulkano::instance::Instance;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

pub trait SubRenderer {
    fn start_render(&self, image: usize); //Start render on gpus, cpu does nothing here

    fn finish_render(
        &self,
        instance: Arc<Instance>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        pool: CpuBufferPool<u16>,
        cpy_image: Arc<StorageImage<vulkano::format::R16G16B16A16Unorm>>,
        image: usize,
    ) -> Box<dyn GpuFuture>;
}

use crate::*;
use crate::{VoxelNode, VoxelTree};
use cgmath::prelude::One;
use cgmath::Quaternion;
use cgmath::Vector3;

#[derive(Clone)]
pub struct VoxelScene {
    pub tree: VoxelTree,
    pub root: u32,
    pub camera_position: Vector3<f64>,
    pub camera_rotation: Quaternion<f64>,
}

impl VoxelScene {
    pub fn empty() -> Self {
        let tree = allocate(11);
        VoxelScene {
            tree,
            root: tree.allocate_node(),
            camera_position: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            camera_rotation: Quaternion::one(),
        }
    }
}

pub mod cpu {

    struct RayTarget<'a> {
        distance: f64,
        node: &'a VoxelNode,
        nodelocation: Vector3<f64>,
        hitpos: Vector3<f64>, //relative to the omnicube
    }

    struct RayResult {
        colour: Vector3<f64>,
        emission: f64,
        metalness: f64,
        roughness: f64,
        hitlocation: Vector3<f64>,
        hitnormal: Vector3<f64>,
        dead: usize,
    }

    use crate::render::*;
    use vulkano::buffer::CpuBufferPool;
    use vulkano::device::Device;
    use vulkano::image::AttachmentImage;
    use vulkano::instance::Instance;
    use vulkano::sync::GpuFuture;

    use cgmath::{Deg, Euler, InnerSpace, Matrix4, Quaternion, Vector3, Vector4};

    use std::sync::Arc;

    const EPSILON: f64 = 0.000002;
    const EMISSION: f64 = 15.0;

    pub struct CpuRenderer {
        width: usize,
        height: usize,
        rpp: usize,
        rbc: usize,
        imagecount: usize,
        pub scenes: Vec<VoxelScene>,
    }

    impl CpuRenderer {
        pub fn new(
            width: usize,
            height: usize,
            rpp: usize,
            rbc: usize,
            imagecount: usize,
        ) -> CpuRenderer {
            CpuRenderer {
                width,
                height,
                rpp,
                rbc,
                imagecount,
                scenes: vec![VoxelScene::empty(); imagecount],
            }
        }

        fn cast_ray_v_box(
            start: Vector3<f64>,
            dir: Vector3<f64>,
            size: f64,
        ) -> (f64, Vector3<f64>) {
            let t1 = (size * -0.5 - start.x) / dir.x;
            let t2 = (size * 0.5 - start.x) / dir.x;

            let t3 = (size * -0.5 - start.y) / dir.y;
            let t4 = (size * 0.5 - start.y) / dir.y;

            let t5 = (size * -0.5 - start.z) / dir.z;
            let t6 = (size * 0.5 - start.z) / dir.z;

            #[inline(always)]
            fn inbox(x: f64, y: f64, z: f64, size: f64) -> bool {
                return x + EPSILON >= size * -0.5
                    && x - EPSILON <= size * 0.5
                    && y + EPSILON >= size * -0.5
                    && y - EPSILON <= size * 0.5
                    && z + EPSILON >= size * -0.5
                    && z - EPSILON <= size * 0.5;
            }

            let pos1 = start + dir * t1;
            let pos2 = start + dir * t2;
            let pos3 = start + dir * t3;
            let pos4 = start + dir * t4;
            let pos5 = start + dir * t5;
            let pos6 = start + dir * t6;

            let mut b1 = (1.0 - (inbox(pos1.x, pos1.y, pos1.z, size) && t1 > 0.0) as i64 as f64)
                * std::f64::MAX;
            let mut b2 = (1.0 - (inbox(pos2.x, pos2.y, pos2.z, size) && t2 > 0.0) as i64 as f64)
                * std::f64::MAX;
            let mut b3 = (1.0 - (inbox(pos3.x, pos3.y, pos3.z, size) && t3 > 0.0) as i64 as f64)
                * std::f64::MAX;
            let mut b4 = (1.0 - (inbox(pos4.x, pos4.y, pos4.z, size) && t4 > 0.0) as i64 as f64)
                * std::f64::MAX;
            let mut b5 = (1.0 - (inbox(pos5.x, pos5.y, pos5.z, size) && t5 > 0.0) as i64 as f64)
                * std::f64::MAX;
            let mut b6 = (1.0 - (inbox(pos6.x, pos6.y, pos6.z, size) && t6 > 0.0) as i64 as f64)
                * std::f64::MAX;

            b1 = b1.max(t1);
            b2 = b2.max(t2);
            b3 = b3.max(t3);
            b4 = b4.max(t4);
            b5 = b5.max(t5);
            b6 = b6.max(t6);

            let min = b1.min(b2.min(b3.min(b4.min(b5.min(b6)))));

            (
                min,
                (pos1 * (b1 == min) as u64 as f64)
                    + (pos2 * (b2 == min) as u64 as f64)
                    + (pos3 * (b3 == min) as u64 as f64)
                    + (pos4 * (b4 == min) as u64 as f64)
                    + (pos5 * (b5 == min) as u64 as f64)
                    + (pos6 * (b6 == min) as u64 as f64),
            )
        }

        fn dead_ray() -> RayResult {
            RayResult {
                colour: new_vec(0.0, 0.0, 0.0),
                emission: 0.0,
                metalness: 0.0,
                roughness: 0.0,
                hitlocation: new_vec(0.0, 0.0, 0.0),
                hitnormal: new_vec(0.0, 0.0, 0.0),
                dead: 0,
            }
        }

        fn cast_ray_voxel<'a>(
            start: Vector3<f64>,
            dir: Vector3<f64>,
            root: &VoxelNode,
            tree: VoxelTree,
        ) -> RayResult {
            let mut size: f64 = 1.0;
            let mut activenodes: Vec<&VoxelNode> = Vec::new();
            let mut nodelocations: Vec<Vector3<f64>> = Vec::new();
            activenodes.push(root);
            nodelocations.push(new_vec(0.0, 0.0, 0.0));

            let mut closestleaf = (std::f64::MAX, None);

            let mut x = 0;
            while x < activenodes.len() {
                let mut targets: Vec<RayTarget> = Vec::new();
                for i in 0..activenodes.len() {
                    let (distance, pos) =
                        CpuRenderer::cast_ray_v_box(start - nodelocations[i], dir, size);
                    targets.push(RayTarget {
                        distance,
                        node: activenodes[i],
                        nodelocation: nodelocations[i],
                        hitpos: pos,
                    });

                    x += 1;
                }
                targets.sort_by(|a, b| (a.distance).partial_cmp(&b.distance).unwrap());
                for h in 0..targets.len() {
                    let hit = &targets[h];
                    if (hit.distance.is_normal() && hit.distance < std::f64::MAX) {
                        if hit.node.flags & 1 != 0 {
                            //leaf
                            if hit.distance < closestleaf.0 {
                                closestleaf = (
                                    hit.distance,
                                    Some(RayResult {
                                        colour: new_vec(
                                            hit.node.colour[0] as f64,
                                            hit.node.colour[1] as f64,
                                            hit.node.colour[2] as f64,
                                        ) / 255.0,
                                        emission: hit.node.emission as f64 / 255.0,
                                        metalness: hit.node.metalness as f64 / 255.0,
                                        roughness: hit.node.roughness as f64 / 255.0,
                                        hitlocation: hit.nodelocation + hit.hitpos,
                                        hitnormal: get_biggest_axis(hit.hitpos).normalize(),
                                        dead: 1,
                                    }),
                                );
                            }
                        } else {
                            for c in 0..8 {
                                let child: u8 = 1 << c;

                                if hit.node.childmask & child != 0 {
                                    nodelocations.push(
                                        hit.nodelocation + (oct_byte_to_vec(child) * size / 2.0),
                                    );
                                    activenodes.push(hit.node.get_child(tree, c));
                                }
                            }
                        }
                    }
                }

                activenodes.drain(0..x);
                nodelocations.drain(0..x);
                x = 0;
                size /= 2.0;
            }

            match closestleaf.1 {
                Some(c) => return c,
                None => return CpuRenderer::dead_ray(),
            }
        }

        fn cast_pixel(
            &self,
            fxmin: f64,
            fxmax: f64,
            fymin: f64,
            fymax: f64,
            campos: Vector3<f64>,
            camrot: Quaternion<f64>,
            tree: VoxelTree,
            root: &VoxelNode,
        ) -> [u16; 4] {
            fn do_ray(
                dir: Vector3<f64>,
                start: Vector3<f64>,
                bouncesleft: usize,
                tree: VoxelTree,
                root: &VoxelNode,
            ) -> Vector3<f64> {
                if (bouncesleft <= 0) {
                    return new_vec(0.0, 0.0, 0.0);
                }

                let cast1 = CpuRenderer::cast_ray_voxel(start, dir, root, tree);

                if cast1.dead == 0 {
                    return new_vec(0.0, 0.0, 0.0);
                }

                let fresnel = fresnel(-dir, cast1.hitnormal) * (1.0 - cast1.roughness);

                let specular =
                    fresnel * ((0.5 * (1.0 - cast1.metalness)) + (0.98 * cast1.metalness));

                let specular_highlight = ((new_vec(0.5, 0.5, 0.5) * (1.0 - cast1.metalness))
                    + (cast1.colour * cast1.metalness))
                    * specular;

                let diffuse = 0.5 * (1.0 - cast1.metalness);

                let newdir = rand_dir_from_surf(cast1.hitnormal);

                let mut combinecolour: Vector3<f64> = new_vec(0.0, 0.0, 0.0);

                if rand::random::<f64>() > (0.5 * (1.0 - cast1.metalness)) {
                    let specular_colour = (1.0 - cast1.emission)
                        * point_wise_mul(
                            specular_highlight,
                            do_ray(
                                (newdir * cast1.roughness)
                                    + (reflect(dir, cast1.hitnormal) * (1.0 - cast1.roughness)),
                                cast1.hitlocation,
                                bouncesleft - 1,
                                tree,
                                root,
                            ),
                        );
                    combinecolour += specular_colour;
                } else {
                    let diffuse_colour = (1.0 - cast1.emission)
                        * (diffuse
                            * point_wise_mul(
                                cast1.colour,
                                do_ray(newdir, cast1.hitlocation, bouncesleft - 1, tree, root),
                            ));
                    combinecolour += diffuse_colour;
                }
                let emissive_colour = (cast1.colour * cast1.emission * EMISSION);

                emissive_colour + combinecolour
            }

            let mut colour: Vector3<f64> = new_vec(0.0, 0.0, 0.0);
            let mut count = 0;

            for _ in 0..self.rpp {
                let raydir = camrot
                    * new_vec(
                        (random_mix(fxmin, fxmax) * 2.0 - 1.0),
                        (random_mix(fymin, fymax) * 2.0 - 1.0),
                        1.0,
                    );
                count += 1;
                colour += do_ray(raydir, campos, self.rbc, tree, root);
            }

            colour /= count as f64;

            let alpha = colour.x.max(colour.y.max(colour.z)) / EMISSION;
            colour /= alpha;

            colour *= std::u16::MAX as f64;

            [
                colour.x as u16,
                colour.y as u16,
                colour.z as u16,
                (alpha * std::u16::MAX as f64) as u16,
            ]

            //return linear_to_srgb(colour);
        }

        pub fn DoRender(&self) {}
    }

    use vulkano::image::StorageImage;

    impl SubRenderer for CpuRenderer {
        fn start_render(&self, image: usize) {}

        fn finish_render(
            &self,
            instance: Arc<Instance>,
            device: Arc<Device>,
            queue: Arc<Queue>,
            pool: CpuBufferPool<u16>,
            cpy_image: Arc<StorageImage<vulkano::format::R16G16B16A16Unorm>>,
            image: usize,
        ) -> Box<dyn GpuFuture> {
            let mut buffer: Vec<u16> = vec![0; (self.width * self.height * 4) as usize]; // Generate the image data;

            let startcpu = std::time::Instant::now();

            use rayon::prelude::*;
            use std::sync::Mutex;

            let columnsdone = Mutex::new(0 as usize);

            let maxindex: usize = self.width * self.height;
            let chunksize: usize = 1000;

            (0..(maxindex / chunksize)).into_par_iter().for_each(|w| {
                for s in 0..(chunksize.min(maxindex.saturating_sub(w * chunksize))) {
                    let x = ((w * chunksize) + s) % self.width;
                    let y = (((w * chunksize) + s) - x) / self.width;
                    let pixelwidth = 1.0 / self.width as f64;
                    let fx = x as f64 * pixelwidth;
                    let pixelheight = 1.0 / self.height as f64;
                    let fy = y as f64 * pixelheight;

                    let hw = pixelwidth / 2.0;
                    let hh = pixelheight / 2.0;

                    let index = (((w * chunksize) + s) * 4 as usize);
                    let colour: [u16; 4] = self.cast_pixel(
                        fx - hw,
                        fx + hw,
                        fy - hh,
                        fy + hh,
                        self.scenes[0].camera_position,
                        self.scenes[0].camera_rotation,
                        self.scenes[0].tree,
                        self.scenes[0].tree.get_node(self.scenes[0].root),
                    );
                    let bufferptr = buffer.as_ptr();
                    unsafe {
                        *(bufferptr.offset(index as isize) as *mut u16) = colour[0];
                        *(bufferptr.offset((index + 1) as isize) as *mut u16) = colour[1];
                        *(bufferptr.offset((index + 2) as isize) as *mut u16) = colour[2];
                        *(bufferptr.offset((index + 3) as isize) as *mut u16) = colour[3];
                    }
                }
                let mut percent = columnsdone.lock().unwrap();
                *percent += 1;
                println!(
                    "Cpu Render is {} % done",
                    ((*percent * chunksize as usize) as f64 * 100.0) / maxindex as f64
                );
            });

            let cpuend = std::time::Instant::now();

            println!("Cpu took {} ms", (cpuend - startcpu).as_millis());

            // Save the buffer as "image.png"
            image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(
                self.width as u32,
                self.height as u32,
                &buffer[..],
            )
            .unwrap()
            .save("image.png")
            .unwrap();

            use vulkano::format::Format;
            use vulkano::image::ImageUsage;
            use vulkano::buffer::BufferUsage;

            let gbuffer = pool.chunk(buffer.clone().into_iter()).unwrap();

            use vulkano::buffer::CpuAccessibleBuffer;

            let mut _usage = BufferUsage::none();
            _usage.transfer_destination = true;

            let gbuffer2 = CpuAccessibleBuffer::from_iter(device.clone(), _usage, false, buffer.into_iter()).unwrap();


            use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};

            let future = AutoCommandBufferBuilder::new(device.clone(), queue.family())
                .unwrap()
                .copy_buffer_to_image(gbuffer.clone(), cpy_image.clone())
                .unwrap()
                .copy_image_to_buffer(cpy_image.clone(), gbuffer2.clone())
                .unwrap()
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();

            future.wait(None).unwrap(); ///////// NOOOTEEE BUG IS 100% not here


            use image::ImageBuffer;
            use image::Rgba;

            queue.wait().unwrap();

            {
                let buffer_content = gbuffer2.read().unwrap();
                let image =
                    ImageBuffer::<Rgba<u16>, _>::from_raw(self.width as u32, self.height as u32, &buffer_content[..])
                    .unwrap();
                image.save("image3.png").unwrap();
            }

            std::thread::sleep_ms(10000);

            Box::new(future)
        }
    }
}
