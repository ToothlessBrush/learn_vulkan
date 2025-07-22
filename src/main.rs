use std::sync::Arc;

use image::ImageBuffer;
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
        CopyImageToBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    format::{ClearColorValue, Format},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{self, GpuFuture},
};

fn main() {
    let renderer = ComputeRenderer::default();
    renderer.run_compute_shader();
}

pub struct ComputeRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl Default for ComputeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeRenderer {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate physical device")
            .next()
            .expect("no physical device available");

        let limits = physical_device.properties().max_image_dimension2_d;
        println!("Max 2D image extent: {limits} x {limits}");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, p)| p.queue_flags.contains(QueueFlags::GRAPHICS))
            .expect("could not find a graphical queue family")
            as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn run_compute_shader(&self) {
        let width = 16384;
        let height = 16384;

        let shader = mandlebrot::load(self.device.clone()).expect("failed to create shader module");
        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();

        let compute_pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [width, height, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let view = ImageView::new_default(image.clone()).unwrap();
        let layout = compute_pipeline.layout();
        let descriptor_set_layout = layout.set_layouts().first().unwrap();

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::image_view(0, view.clone())],
            [],
        )
        .unwrap();

        let output_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..width * height * 4).map(|_| 0u8),
        )
        .expect("failed to create output buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .dispatch([width / 8, height / 8, 1])
                .unwrap()
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    image.clone(),
                    output_buffer.clone(),
                ))
                .unwrap();
        }

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let buffer_content = output_buffer.read().unwrap();
        let image = ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, &buffer_content[..])
            .unwrap();
        image.save("image.png").unwrap();

        println!("Image saved successfully.");
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

mod mandlebrot {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            // Mandelbrot configuration
            const float max_iter = 500.0;
            const float zoom = 2.5;
            const vec2 center = vec2(-0.5, 0.0);

            void main() {
                ivec2 size = imageSize(img);
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(size);
                vec2 scaled = (norm_coordinates - vec2(0.5)) * zoom + center;

                vec2 z = vec2(0.0);
                float i;
                for (i = 0.0; i < max_iter; i++) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + scaled.x,
                        2.0 * z.x * z.y + scaled.y
                    );
                    if (length(z) > 4.0)
                        break;
                }

                float smooth_i = i - log2(log(length(z))) + 4.0;
                float t = clamp(smooth_i / max_iter, 0.0, 1.0);

                // Blue outside, black inside
                vec3 color = mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.4 + 0.6 * t, 1.0), t);

                imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
            }
        "
    }
}
