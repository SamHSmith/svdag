use lib::*;
use antelope::cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
use antelope::camera::RenderCamera;
use antelope::mesh::{Mesh,MeshCreateInfo, MeshFactory, PostVertex, RenderInfo, Vertex};
use antelope::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
use antelope::{MeshRenderer, MeshFrame};


fn main() {
    unsafe {
        println!("Hello voxels");

        let tree = allocate(1000);
        let node = VoxelNode::new(&tree);
        (*node).put_child(0,node);
        (*node).put_child(1,node);
        (*node).put_child(7,node);
        (*node).put_child(6,node);

        let mut mats = Vec::new();
        for i in 0..8{
            let child : u8 = 1 << i;
            if (child & (*node).childmask) != 0 {
                let (x,y,z) = oct_byte_to_vec(child);
                println!("{} {} {}", x,y,z);
                mats.push(Matrix4::from_translation(Vector3 {
                    x,
                    y,
                    z,
                }));
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
        let mut angle : f64 = 0.0;

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
            angle+=0.1;
            std::thread::sleep_ms(200);
        }

        thread.join().ok().unwrap();
    }
}
