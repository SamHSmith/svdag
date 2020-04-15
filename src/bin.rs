use antelope::camera::RenderCamera;
use antelope::cgmath::prelude::One;
use antelope::cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
use antelope::mesh::{Mesh, MeshCreateInfo, MeshFactory, PostVertex, RenderInfo, Vertex};
use antelope::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
use antelope::{MeshFrame, MeshRenderer};
use lib::*;

const EPSILON: f64 = 0.0001;

fn cast_ray_v_box(start: Vector3<f64>, dir: Vector3<f64>, size: f64) -> bool {
    let t1 = (size * -0.5 - start.x) / dir.x;
    let t2 = (size * 0.5 - start.x) / dir.x;

    let t3 = (size * -0.5 - start.y) / dir.y;
    let t4 = (size * 0.5 - start.y) / dir.y;

    let t5 = (size * -0.5 - start.z) / dir.z;
    let t6 = (size * 0.5 - start.z) / dir.z;

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

    inbox(pos1.x, pos1.y, pos1.z, size)
        || inbox(pos2.x, pos2.y, pos2.z, size)
        || inbox(pos3.x, pos3.y, pos3.z, size)
        || inbox(pos4.x, pos4.y, pos4.z, size)
        || inbox(pos5.x, pos5.y, pos5.z, size)
        || inbox(pos6.x, pos6.y, pos6.z, size)
}

fn cast_ray_voxel(
    start: Vector3<f64>,
    dir: Vector3<f64>,
    root: &VoxelNode,
    tree: VoxelTree,
) -> [u8; 3] {
    let mut size: f64 = 1.0;
    let mut activenodes: Vec<&VoxelNode> = Vec::new();
    let mut nodelocations: Vec<Vector3<f64>> = Vec::new();
    activenodes.push(root);
    nodelocations.push(new_vec(0.0, 0.0, 0.0));

    let mut x = 0;
    while x < activenodes.len() {
        for i in 0..activenodes.len() {
            if cast_ray_v_box(start - nodelocations[i], dir, size) {
                if activenodes[i].flags & 1 != 0 {
                    //leaf
                    return activenodes[i].colour;
                } else {
                    for c in 0..8 {
                        let child: u8 = 1 << c;

                        if activenodes[i].childmask & child != 0 {
                            nodelocations
                                .push(nodelocations[i] + (oct_byte_to_vec(child) * size / 2.0));
                            activenodes.push(activenodes[i].get_child(tree, c));
                        }
                    }
                }
            }
            x += 1;
        }
        activenodes.drain(0..x);
        nodelocations.drain(0..x);
        x = 0;
        size /= 2.0;
    }

    return [0, 0, 0];
}

fn new_vec(x: f64, y: f64, z: f64) -> Vector3<f64> {
    return Vector3 { x, y, z };
}

fn main() {
    let mut tree = allocate(500);
    let node: &mut VoxelNode = tree.allocate_and_get_node();
    node.put_child(0, tree.allocate_node());
    let node2: &mut VoxelNode = node.get_child(tree, 0);
    node2.put_child(0, tree.allocate_node());
    node2.get_child(tree, 0).flags = 1;
    node2.get_child(tree, 0).colour = [230, 183, 0];
    node2.put_child(6, tree.allocate_node());
    node2.get_child(tree, 6).flags = 1;
    node2.get_child(tree, 6).colour = [0, 183, 235];
    node2.put_child(1, tree.allocate_node());
    node2.get_child(tree, 1).flags = 1;
    node2.get_child(tree, 1).colour = [200, 183, 235];

    let width: u32 = 1024;
    let height: u32 = 1024;

    let campos = new_vec(2.0, -1.5, -3.5);
    let camrot = new_vec(-20.0, -20.0, 0.0);
    let rotation = Quaternion::from(Euler {
        x: Deg(camrot.x),
        y: Deg(camrot.y),
        z: Deg(camrot.z),
    });

    let cubepos = new_vec(0.0, 1.0, 0.0);

    let mut buffer: Vec<u8> = vec![0; (width * height * 3) as usize]; // Generate the image data;

    for x in 0..width {
        for y in 0..height {
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            let index = ((y * width + x) * 3 as u32) as usize;
            let colour: [u8; 3] = cast_ray_voxel(
                campos - cubepos,
                rotation * new_vec((fx * 2.0 - 1.0), (fy * 2.0 - 1.0), 1.0),
                node,
                tree,
            );

            buffer[index] = colour[0];
            buffer[index + 1] = colour[1];
            buffer[index + 2] = colour[2];
        }
    }

    // Save the buffer as "image.png"
    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, &buffer[..])
        .unwrap()
        .save("image.png")
        .unwrap();

    use image::ImageBuffer;
    use image::Rgba;
    use std::sync::Arc;
    use vulkano::buffer::BufferUsage;
    use vulkano::buffer::CpuAccessibleBuffer;
    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
    use vulkano::device::Device;
    use vulkano::device::DeviceExtensions;
    use vulkano::device::Features;
    use vulkano::format::Format;
    use vulkano::image::Dimensions;
    use vulkano::image::StorageImage;
    use vulkano::instance::Instance;
    use vulkano::instance::InstanceExtensions;
    use vulkano::instance::PhysicalDevice;
    use vulkano::pipeline::ComputePipeline;
    use vulkano::sync::GpuFuture;

    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let mut feat = Features::none();
    feat.shader_f3264 = true;

    let (device, mut queues) = {
        Device::new(
            physical,
            &feat,
            &DeviceExtensions::none(),
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: 1024,
            height: 1024,
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
    .unwrap();

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

layout(push_constant) uniform pushConstants {
    dvec3 campos;
    dvec3 right;
    dvec3 up;
    dvec3 forward;
} pc;

const double EPSILON = 0.0001;

bool inbox(dvec3 pos){
    return (pos.x + EPSILON >= -0.5) && (pos.x - EPSILON <= 0.5) && (pos.y + EPSILON >= -0.5) && (pos.y - EPSILON <= 0.5) && (pos.z + EPSILON >= -0.5) && (pos.z - EPSILON <= 0.5);
}

bool cast_ray(dvec3 start, dvec3 dir) {
    double t1 = (-0.5 - start.x) / dir.x;
    double t2 = (0.5 - start.x) / dir.x;

    double t3 = (-0.5 - start.y) / dir.y;
    double t4 = (0.5 - start.y) / dir.y;

    double t5 = (-0.5 - start.z) / dir.z;
    double t6 = (0.5 - start.z) / dir.z;

    dvec3 pos1 = start + (dir * t1);
    dvec3 pos2 = start + (dir * t2);
    dvec3 pos3 = start + (dir * t3);
    dvec3 pos4 = start + (dir * t4);
    dvec3 pos5 = start + (dir * t5);
    dvec3 pos6 = start + (dir * t6);



    return inbox(pos1)
        || inbox(pos2)
        || inbox(pos3)
        || inbox(pos4)
        || inbox(pos5)
        || inbox(pos6)
;
}

void main() {
    dvec2 coords = dvec2(gl_GlobalInvocationID.xy) / dvec2(imageSize(img));

    dvec3 start =  pc.campos;
    dvec3 ray =  ((coords.x * 2.0 - 1.0) * pc.right) + ((coords.y * 2.0 - 1.0) * pc.up) + pc.forward;

/// cyan vec3(0.0, 183.0 / 255.0, 235.0 / 255.0)
    vec4 to_write = vec4(0.0,0.0,0.0,1.0);

    if(cast_ray(start, ray)){
        to_write = vec4(vec3(0.0, 183.0 / 255.0, 235.0 / 255.0), 1.0);
    }

    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}"
        }
    }

    let shader = cs::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let set = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap()
                .clone(),
        )
        .add_image(image.clone())
        .unwrap()
        .build()
        .unwrap(),
    );

    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let right = rotation
        * Vector3::<f64> {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
    let up = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
    let forward = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

    let push_constants = cs::ty::pushConstants {
        campos: [campos.x, campos.y, campos.z],
        right: [right.x, right.y, right.z],
        up: [up.x, up.y, up.z],
        forward: [forward.x, forward.y, forward.z],
        _dummy0: [0; 8],
        _dummy1: [0; 8],
        _dummy2: [0; 8],
    };

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [1024 / 8, 1024 / 8, 1],
            compute_pipeline.clone(),
            set.clone(),
            push_constants,
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image2.png").unwrap();

    return;

    println!("Hello voxels");

    let tree = allocate(1000);
    let node = tree.get_node(tree.allocate_node());
    node.put_child(0, 1);
    node.put_child(1, 2);
    node.put_child(7, 3);
    node.put_child(6, 4);

    let mut mats = Vec::new();
    for i in 0..8 {
        let child: u8 = 1 << i;
        if (child & (*node).childmask) != 0 {
            let xyz = oct_byte_to_vec(child);
            println!("{:?}", xyz);
            mats.push(Matrix4::from_translation(xyz));
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
