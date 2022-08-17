mod texture;

use std::{fs::File, io::Read, borrow::Cow};

use bytemuck_derive::{Pod, Zeroable};
use cgmath::Matrix4;
use wgpu::{Instance, Backends, RequestAdapterOptions, PowerPreference, DeviceDescriptor, Device, Queue, BufferUsages, VertexBufferLayout, VertexAttribute, Buffer, ShaderModuleDescriptor, ShaderSource, RenderPipelineDescriptor, FragmentState, VertexState, MultisampleState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, FrontFace, Face, PolygonMode, RenderPipeline, Surface, ColorTargetState, ColorWrites, BlendState, SurfaceConfiguration, PresentMode, TextureUsages, RenderPassDescriptor, RenderPassColorAttachment, Operations, Color, CommandEncoderDescriptor, VertexStepMode, VertexFormat, BufferDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType, BindGroupDescriptor, BindGroupEntry, BindGroup, DepthStencilState, CompareFunction, StencilState, DepthBiasState, RenderPassDepthStencilAttachment, LoadOp, TextureSampleType, TextureDimension, SamplerBindingType, TextureViewDimension, BindingResource};
use winit::{window::Window, dpi::PhysicalSize};

use crate::{world::{terrain::WorldVertex}, ArcWorld, renderer::texture::Texture};

pub struct Renderer {
  surface: Surface,
  surface_cfg: SurfaceConfiguration,
  device: Device,
  queue: Queue,
  vertex_buffer: Buffer,
  vertex_buffer_items: u64, //Length in items (not bytes)
  vertex_buffer_limit: u64, //Also in items
  main_tex_bind_group: BindGroup,
  camera_buffer: Buffer,
  camera_bind_group: BindGroup,
  depth_texture: Texture,
  pipeline: RenderPipeline,
  size: PhysicalSize<u32>,
  world: ArcWorld
}

const VERTEX_BUFFER_SPARE: u64 = 10000; //This is items, not bytes.

impl Renderer {
  pub async fn new(window: &Window, world: ArcWorld) -> Result<Self, RendererCreateError> {
    let size = window.inner_size();
    let instance = Instance::new(Backends::PRIMARY);
    let surface = unsafe {instance.create_surface(window) };

    let adapter = instance.request_adapter(&RequestAdapterOptions {
      power_preference: PowerPreference::HighPerformance,
      ..Default::default()
    }).await.ok_or(RendererCreateError::NoDeviceFound)?;

    
    let surface_cfg = SurfaceConfiguration {
      format: surface.get_supported_formats(&adapter)[0],
      height: size.height,
      width: size.width,
      present_mode: PresentMode::Fifo,
      usage: TextureUsages::RENDER_ATTACHMENT
    };
    
    println!("Using {}", adapter.get_info().name);
    

    let (device, queue) = adapter.request_device(&DeviceDescriptor {
      ..Default::default() //TODO fix.
    }, None).await.map_err(|_| RendererCreateError::RequestDeviceError)?;

    let vertex_buffer = device.create_buffer(&BufferDescriptor {
      label: Some("Very cool vertex buffer."),
      mapped_at_creation: false,
      size: VERTEX_BUFFER_SPARE * std::mem::size_of::<WorldVertex>() as u64,
      usage: BufferUsages::VERTEX | BufferUsages::COPY_DST
    });

    let vertex_buffer_items: u64 = 0;
    let vertex_buffer_limit: u64 = VERTEX_BUFFER_SPARE;

    let mut shader_file = File::open("./shaders/shader.wgsl").map_err(|_| RendererCreateError::ShaderLoadError)?;
    let mut shader: String = String::new();
    shader_file.read_to_string(&mut shader).map_err(|_| RendererCreateError::ShaderLoadError)?;

    let shader_module = device.create_shader_module(ShaderModuleDescriptor { 
      label: Some("very cool shader module"), 
      source: ShaderSource::Wgsl(Cow::from(shader))
    });

    let camera_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("camera buffer and stuff"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
      entries: &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
      label: Some("camera bind group layout"), 
    });

    let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("camera bind group stuff"),
        layout: &camera_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let mut image_bytes = Vec::new();
    File::open("./texture.png").expect("Failed to open main texture!").read_to_end(&mut image_bytes).expect("Image file had a read error.");
    let img = image::load_from_memory(&image_bytes).expect("Main texture is corrupt.");
    
    let main_texture = Texture::from_image(&device, &queue, &img, Some("VERY COOL IMAGE")).unwrap();

    let main_tex_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
          BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
          },
          BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None
          }
        ],
    });

    let main_tex_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &main_tex_layout,
        entries: &[
          BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&main_texture.view),
          },
          BindGroupEntry {
            binding: 1,
            resource: BindingResource::Sampler(&main_texture.sampler)
          }
        ],
    });

    let depth_texture = Texture::create_depth_texture(&device, &surface_cfg, "Depth texture and stuff");

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[&camera_bind_group_layout, &main_tex_layout],
      label: Some("stinky pipeline layout"),
      push_constant_ranges: &[]
    }); //Not really necessary right now.

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
      depth_stencil: Some(DepthStencilState {
        format: Texture::DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    }),
      vertex: VertexState {
        buffers: &[WorldVertex::desc()],
        entry_point: "vs_main",
        module: &shader_module
      },
      fragment: Some(FragmentState {
        entry_point: "fs_main",
        module: &shader_module,
        targets: &[Some(ColorTargetState {
          blend: Some(BlendState::REPLACE),
          format: surface_cfg.format,
          write_mask: ColorWrites::ALL
        })]
      }),
      label: Some("Very cool pipeline layout."),
      layout: Some(&pipeline_layout),
      multisample: MultisampleState {
        count: 1, //MSAA?
        mask: !0,
        alpha_to_coverage_enabled: false, //AA
    },
      multiview: None,
      primitive: PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: Some(Face::Back),
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
      },
    });

    Ok(Self {
      surface,
      surface_cfg,
      device,
      queue,
      vertex_buffer,
      vertex_buffer_items,
      vertex_buffer_limit,
      main_tex_bind_group,
      depth_texture,
      camera_bind_group,
      camera_buffer,
      pipeline,
      size,
      world
    })
  }

  pub fn resize(&mut self, size: PhysicalSize<u32>) {
    if size.width > 32 && size.height > 32 {
      self.size = size;
      self.surface_cfg.width = size.width;
      self.surface_cfg.height = size.height;
      self.surface.configure(&self.device, &self.surface_cfg);
      self.depth_texture = Texture::create_depth_texture(&self.device, &self.surface_cfg, "another depth texture");
    }
  }

  fn update_world_vertices(&mut self) {
    let verts;
    let mut world_terrain_update = self.world.lock().unwrap();
    if let Some(new_verts) = world_terrain_update.get_terrain_vertex_update() {
      verts = new_verts;
    } else {
      return;
    }

    if verts.len() as u64 > self.vertex_buffer_limit { //Recreate buffer.
      self.vertex_buffer.destroy();

      let new_limit = verts.len() as u64 + VERTEX_BUFFER_SPARE;

      let new_buffer = self.device.create_buffer(&BufferDescriptor {
        label: Some("cool vertex buffer 2"),
        mapped_at_creation: false,
        size:  new_limit * std::mem::size_of::<WorldVertex>() as u64,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST, //Add some spare space too for more vertices.
      });

      self.vertex_buffer = new_buffer;
      self.vertex_buffer_limit = new_limit;
    }
    self.vertex_buffer_items = verts.len() as u64; //Update length.
    self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(verts)); //Write buffer. Assuming that the buffer will be written on submit() on the next render call.
  }

  pub fn render(&mut self) -> Result<(), RenderError> {
    self.update_world_vertices();

    let world = self.world.lock().unwrap();
    let view_mat = world.get_player_view(self.size.width as f32/self.size.height as f32);
    let camera_view = CameraUniform::from(view_mat);
    self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_view]));

    let out = self.surface.get_current_texture().map_err(|_| RenderError::SurfaceError)?;
    let view = out.texture.create_view(&Default::default());

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("very cool command encoder")
    });

    if self.vertex_buffer_items > 0 {
      let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            load: LoadOp::Clear(Color {
              r: 0.0,
              g: 0.0,
              b: 0.5,
              a: 0.0
            }),
            store: true
          },
          resolve_target: None,
          view: &view
        })],
        label: Some("render pass and stuff"),
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &self.depth_texture.view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        })
      });

      render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..self.vertex_buffer_items * std::mem::size_of::<WorldVertex>() as u64));
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.main_tex_bind_group, &[]);
      render_pass.set_pipeline(&self.pipeline);
      render_pass.draw(0..self.vertex_buffer_items as u32, 0..1);
    }

    let command_buffers = std::iter::once(encoder.finish());
    self.queue.submit(command_buffers);
    out.present();

    Ok(())
  }
}

impl Drop for Renderer {
  fn drop(&mut self) {
    //self.vertex_buffer.unmap();
    self.vertex_buffer.destroy(); //Drop vertex buffer.
  }
}

trait Descriptable {
  fn desc<'a>() -> VertexBufferLayout<'a>;
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniform {
  view: [[f32; 4]; 4]
}

impl From<Matrix4<f32>> for CameraUniform {
  fn from(m: Matrix4<f32>) -> Self {
      Self {
        view: m.into()
      }
  }
}

impl Descriptable for WorldVertex {
  fn desc<'a>() -> VertexBufferLayout<'a> {
    VertexBufferLayout {
      array_stride: std::mem::size_of::<WorldVertex>() as u64,
      step_mode: VertexStepMode::Vertex,
      attributes: &[
        VertexAttribute { //Position
          format: VertexFormat::Float32x3,
          offset: 0,
          shader_location: 0
        },
        VertexAttribute { //UV Mapping (Future)
          format: VertexFormat::Float32x2,
          offset: std::mem::size_of::<[f32; 3]>() as u64,
          shader_location: 1
        }
      ],
    }
  }
}

#[derive(Debug)]
pub enum RendererCreateError {
  NoDeviceFound,
  RequestDeviceError,
  ShaderLoadError,
}

#[derive(Debug)]
pub enum RenderError {
  SurfaceError
}