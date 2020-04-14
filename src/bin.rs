use antelope::camera::RenderCamera;
use antelope::cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
use antelope::mesh::{Mesh, MeshCreateInfo, MeshFactory, PostVertex, RenderInfo, Vertex};
use antelope::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
use antelope::{MeshFrame, MeshRenderer};
use lib::*;

fn cast_ray(start: Vector3<f64>, dir: Vector3<f64>) -> bool {
    let mut t1 = (0.0 - start.x) / dir.x;
    let mut t2 = (1.0 - start.x) / dir.x;

    let mut t3 = (0.0 - start.y) / dir.y;
    let mut t4 = (1.0 - start.y) / dir.y;

    let mut t5 = (0.0 - start.z) / dir.z;
    let mut t6 = (1.0 - start.z) / dir.z;

    fn inbox(x: f64, y: f64, z: f64) -> bool {
        return x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0 && z >= 0.0 && z <= 1.0;
    }
    let pos1 = start + dir * t1;
    let pos2 = start + dir * t2;
    let pos3 = start + dir * t3;
    let pos4 = start + dir * t4;
    let pos5 = start + dir * t5;
    let pos6 = start + dir * t6;

           inbox(pos1.x, pos1.y, pos1.z)
        || inbox(pos2.x, pos2.y, pos2.z)
        || inbox(pos3.x, pos3.y, pos3.z)
        || inbox(pos4.x, pos4.y, pos4.z)
        || inbox(pos5.x, pos5.y, pos5.z)
        || inbox(pos6.x, pos6.y, pos6.z)
}

fn new_vec(x: f64, y: f64, z: f64) -> Vector3<f64> {
    return Vector3 { x, y, z };
}

fn main() {
    let width: u32 = 512;
    let height: u32 = 512;

    let campos = new_vec(1.5, 0.5, -1.0);
    let camrot = new_vec(0.0,-30.0,0.0);

    let mut buffer: Vec<u8> = vec![0; (width * height * 4) as usize]; // Generate the image data;

    for x in 0..width {
        for y in 0..height {
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;
            let rotation = Quaternion::from(Euler {
                x: Deg(camrot.x),
                y: Deg(camrot.y),
                z: Deg(camrot.z),
            });

            if cast_ray(
                campos,
                rotation * new_vec((fx * 2.0 - 1.0), (fy * 2.0 - 1.0), 1.0),
            ) {
                let index = ((y * width + x) * 4 as u32) as usize;
                let colour : [u8; 3] = [0, 183, 235];

                buffer[index] = colour[0];
                buffer[index + 1] = colour[1];
                buffer[index + 2] = colour[2];
                buffer[index + 3] = 255;
            }
        }
    }

    // Save the buffer as "image.png"
    image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, &buffer[..])
        .unwrap()
        .save("image.png")
        .unwrap();

    return;

    unsafe {
        println!("Hello voxels");

        let tree = allocate(1000);
        let node = VoxelNode::new(&tree);
        (*node).put_child(0, node);
        (*node).put_child(1, node);
        (*node).put_child(7, node);
        (*node).put_child(6, node);

        let mut mats = Vec::new();
        for i in 0..8 {
            let child: u8 = 1 << i;
            if (child & (*node).childmask) != 0 {
                let (x, y, z) = oct_byte_to_vec(child);
                println!("{} {} {}", x, y, z);
                mats.push(Matrix4::from_translation(Vector3 { x, y, z }));
            }
        }

        let (thread, win) = antelope::window::main_loop::<MeshRenderer, MeshFrame>();

        let meshinfo = MeshCreateInfo {
            verticies: vec![
                Vertex {
                    position: [0.0, 0.0, 0.0],
                    colour: [1.0, 0.0, 0.0],
                    normal: [1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 0.0, -0.0],
                    colour: [0.0, 1.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 1.0, -1.0],
                    colour: [0.0, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.0, -0.9],
                    colour: [1.0, 1.0, 1.0],
                    normal: [1.0, 1.0, 1.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
            ],
            indicies: vec![0, 1, 2, 0, 2, 3, 0, 1, 3, 2, 1, 3],
        };

        let mesh = win.mesh_factory.create_mesh(meshinfo);

        let cam = RenderCamera {
            position: Vector3 {
                x: -6.0,
                y: 4.0,
                z: 5.0,
            },
            rotation: Quaternion::from(Euler::new(Deg(30.0), Deg(60.0), Deg(0.0))),
            aspect: 1.0,
            fov: 90.0,
            far: 10000.0,
            near: 0.1,
        };
        let mut angle: f64 = 0.0;

        let mut meshes = Vec::new();
        for x in &mats {
            meshes.push(mesh.clone());
        }
        loop {
            win.render_info.push(RenderInfo {
                meshes: meshes.clone(),
                mats: mats.clone(),
                camera: cam.clone(),
            });
            angle += 0.1;
            std::thread::sleep_ms(200);
        }

        thread.join().ok().unwrap();
    }
}
