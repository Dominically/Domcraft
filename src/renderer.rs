use std::{fs::File, io::Read, borrow::Cow};

use bytemuck_derive::{Pod, Zeroable};
use wgpu::{Instance, Backends, RequestAdapterOptions, PowerPreference, DeviceDescriptor, Device, Queue, util::{DeviceExt, BufferInitDescriptor}, BufferUsages, VertexBufferLayout, VertexAttribute, Buffer, ShaderModuleDescriptor, ShaderSource, RenderPipelineDescriptor, FragmentState, VertexState, MultisampleState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, FrontFace, Face, PolygonMode, RenderPipeline, Surface, ColorTargetState, ColorWrites, BlendState, SurfaceConfiguration, PresentMode, TextureUsages, RenderPassDescriptor, RenderPassColorAttachment, Operations, Color, CommandEncoderDescriptor, VertexStepMode, VertexFormat, BufferDescriptor};
use winit::{window::Window, dpi::PhysicalSize};

use crate::world::{terrain::WorldVertex, World};

pub struct Renderer {
  surface: Surface,
  surface_cfg: SurfaceConfiguration,
  device: Device,
  queue: Queue,
  vertex_buffer: Buffer,
  vertex_buffer_items: u64,
  vertex_buffer_limit: u64,
  pipeline: RenderPipeline,
  size: PhysicalSize<u32>
}

const VERTEX_BUFFER_SPARE: u64 = 10000;

impl Renderer {
  pub async fn new(window: &Window) -> Result<Self, RendererCreateError> {
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
      mapped_at_creation: true,
      size: VERTEX_BUFFER_SPARE * std::mem::size_of::<WorldVertex>() as u64,
      usage: BufferUsages::VERTEX
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

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[],
      label: Some("stinky pipeline layout"),
      push_constant_ranges: &[]
    }); //Not really necessary right now.

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
      depth_stencil: None,
      vertex: VertexState {
        buffers: &[Vertex::desc()],
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
      pipeline,
      size
    })
  }

  pub fn resize(&mut self, size: PhysicalSize<u32>) {
    if size.width > 32 && size.height > 32 {
      self.size = size;
      self.surface_cfg.width = size.width;
      self.surface_cfg.height = size.height;
      self.surface.configure(&self.device, &self.surface_cfg);
    }
  }

  pub fn update_world_vertices(&mut self, verts: Vec<WorldVertex>) {
    todo!();
  }

  pub fn render(&self) -> Result<(), RenderError> {
    let out = self.surface.get_current_texture().map_err(|_| RenderError::SurfaceError)?;
    let view = out.texture.create_view(&Default::default());

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("very cool command encoder")
    });

    if self.vertex_buffer_items > 0 {
      let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
          ops: Operations {
            load: wgpu::LoadOp::Clear(Color {
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
        depth_stencil_attachment: None //No depth stencil for now.
      });

      render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..self.vertex_buffer_items));
      render_pass.set_pipeline(&self.pipeline);
      render_pass.draw(0..self.vertex_buffer_items as u32, 0..1);
    }

    let command_buffers = std::iter::once(encoder.finish());
    self.queue.submit(command_buffers);
    out.present();
    println!("Rendered");

    Ok(())
  }
}

trait Descriptable {
  fn desc<'a>() -> VertexBufferLayout<'a>;
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
  position: [f32; 2] //2D for now.
}

impl Descriptable for Vertex {
  fn desc<'a>() -> VertexBufferLayout<'a> {
    const POSITION_ATTRIBUTE: VertexAttribute = VertexAttribute {
      format: VertexFormat::Float32x2, //TODO change this when I do 3D.
      offset: 0,
      shader_location: 0
    };

    VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &[POSITION_ATTRIBUTE],
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