import bpy
import numpy as np
import math
import mathutils


##############################################################################

# Config variables

# Changes these paths to the paths where:
# 1) outputPath: you want to save the rendered image (demo.png)
# 2) meshPath: the input polygon mesh is located
output_path = 'D:\\work\\code\\blender_render\\demo.png'
input_mesh_path = 'D:\\work\\code\\blender_render\\onsurface.obj'

# Location/rotation/scale are mesh dependent: 
# these values correspond to onsurface.obj, they should be changed for different object.
# Used by the function readOBJ that reads and init the mesh.
location = (0.0, 0.0, 0.750)
rotation = (0, 0, 90)
scale = (0.1, 0.1, 0.1)

# Does the object have sharp features? 
has_sharp_features = False

# Rendering parameters 
image_res_x = 720 # image resolution 
image_res_y = 720 
num_samples = 50 # number of samples per pixel
exposure = 1.0 


##############################################################################


##############################################################################

# Helper functions


def initBlender(resolution_x, resolution_y, num_samples = 128, exposure = 1.5):
	bpy.ops.object.select_all(action = 'SELECT')
	bpy.ops.object.delete() 
	bpy.context.scene.render.engine = 'CYCLES'
	bpy.context.scene.render.resolution_x = resolution_x 
	bpy.context.scene.render.resolution_y = resolution_y 
	bpy.context.scene.cycles.film_transparent = True
	bpy.context.scene.cycles.device = 'GPU'
	bpy.context.scene.cycles.samples = num_samples 
	bpy.context.scene.cycles.max_bounces = 6
	bpy.context.scene.cycles.film_exposure = exposure
	bpy.data.scenes[0].view_layers['View Layer']['cycles']['use_denoising'] = 1


def readOBJ(file_path, location, rotation_euler, scale):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)
	bpy.ops.import_scene.obj(filepath=file_path)
	mesh = bpy.context.selected_objects[0]
	mesh.location = location
	mesh.rotation_euler = angle
	mesh.scale = scale
	return mesh 


class ColorObj:
	def __init__(self, RGBA = (144.0/255, 210.0/255, 236.0/255, 1), \
				 H = 0.5, S = 1.0, V = 1.0,\
				 B = 0.0, C = 0.0):
		self.H = H 
        self.S = S 
        self.V = V 
        self.RGBA = RGBA
        self.B = B 
        self.C = C 


def initColorNode(tree, color):
	HSV = tree.nodes.new('ShaderNodeHueSaturation')
	HSV.inputs['Color'].default_value = color.RGBA
	HSV.inputs['Saturation'].default_value = color.S
	HSV.inputs['Value'].default_value = color.V
	HSV.inputs['Hue'].default_value = color.H
	BS = tree.nodes.new('ShaderNodeBrightContrast')
	BS.inputs['Bright'].default_value = color.B
	BS.inputs['Contrast'].default_value = color.C
	tree.links.new(HSV.outputs['Color'], BS.inputs['Color'])
	return BS


def setMaterial(mesh, mesh_color):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	BCNode = initColorNode(tree, mesh_color)

	fresnel = tree.nodes.new('ShaderNodeFresnel')
	fresnel.inputs[0].default_value = 1.3

	glass = tree.nodes.new('ShaderNodeBsdfGlass')
	tree.links.new(BCNode.outputs['Color'], glass.inputs['Color'])

	transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
	tree.links.new(BCNode.outputs['Color'], transparent.inputs['Color'])

	multiply = tree.nodes.new('ShaderNodeMixRGB')
	multiply.blend_type = 'MULTIPLY'
	multiply.inputs['Fac'].default_value = 1
	multiply.inputs['Color2'].default_value = (1,1,1,1)
	tree.links.new(fresnel.outputs['Fac'], multiply.inputs['Color1'])

	mix1 = tree.nodes.new('ShaderNodeMixShader')
	mix1.inputs['Fac'].default_value = 0.7
	tree.links.new(glass.outputs[0], mix1.inputs[1])
	tree.links.new(transparent.outputs[0], mix1.inputs[2])

	glossy = tree.nodes.new('ShaderNodeBsdfGlossy')
	glossy.inputs['Color'].default_value = (0.8, 0.72, 0.437, 1)

	mix2 = tree.nodes.new('ShaderNodeMixShader')
	tree.links.new(multiply.outputs[0], mix2.inputs[0])
	tree.links.new(mix1.outputs[0], mix2.inputs[1])
	tree.links.new(glossy.outputs[0], mix2.inputs[2])

	tree.links.new(mix2.outputs[0], tree.nodes['Material Output'].inputs['Surface'])


def createInvisibleGround(location = (0,0,0), ground_size = 5, shadow_light = 0.7):
	bpy.context.scene.cycles.film_transparent = True
	bpy.ops.mesh.primitive_plane_add(location = location, size = ground_size)
	bpy.context.object.cycles.is_shadow_catcher = True

	ground = bpy.context.object
	mat = bpy.data.materials.new('MeshMaterial')
	ground.data.materials.append(mat)
	mat.use_nodes = True
	tree = mat.node_tree
	tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = shadow_light


def lookAt(camera, point):
	direction = point - camera.location
	rotQuat = direction.to_track_quat('-Z', 'Y')
	camera.rotation_euler = rotQuat.to_euler()


def setCamera(cam_location, look_at_location = (0,0,0), focal_length = 35):
	bpy.ops.object.camera_add(location = cam_location)
	cam = bpy.context.object
	cam.data.lens = focal_length
	loc = mathutils.Vector(look_at_location)
	lookAt(cam, loc)
	return cam


def setAmbientLight(color = (0,0,0,1)):
	bpy.data.scenes[0].world.use_nodes = True
	bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color


def setSunLight(rotation_euler, strength, shadow_soft_size = 0.05):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)
	bpy.ops.object.light_add(type = 'SUN', rotation = angle)
	lamp = bpy.data.lights['Sun']
	lamp.shadow_soft_size = shadow_soft_size
	
	if bpy.context.scene.render.engine.title() == 'Blender_Eevee':
		lamp.energy = strength
	else:
		lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength

	return lamp


##############################################################################

# Main script

# Init blender
initBlender(image_res_x, image_res_y, num_samples, exposure)

# Read input mesh 
mesh = readOBJ(input_mesh_path, location, rotation, scale)

if has_sharp_features:
	bpy.ops.object.shade_flat()	
else:
	bpy.ops.object.shade_smooth()

# Material
amber = (100/255.0, 75/255.0, 0/255.0, 255/255.0) 
mesh_color = ColorObj(amber, 0.5, 1.0, 1.0, 0.4, 0.0)
setMaterial(mesh, mesh_color)

# For shadows 
ground_center = (0, 0, 0)
shadow_darkness = 0.7
ground_size = 20
createInvisibleGround(ground_center, ground_size, shadow_darkness)

# Camera information 
# No need to touch this since the mesh should be rescaled properly
cam_location = (1.9, 2, 2.2)
look_at_location = (0, 0, 0.5)
focal_length = 45
cam = setCamera(cam_location, look_at_location, focal_length)

# Directional light
light_angle = (-15, -34, -155) 
strength = 2
shadow_softness = 0.1
sun = setSunLight(light_angle, strength, shadow_softness)

# Ambient light
ambient_color = (0.2, 0.2, 0.2, 1)
setAmbientLight(ambient_color)

# Render and save 
# To run this script from the command line: 
# $ blender --background --python blender_render_demo.py
bpy.data.scenes['Scene'].render.filepath = output_path
bpy.data.scenes['Scene'].camera = cam
bpy.ops.render.render(write_still = True)

# reset everything
# bpy.wm.read_homefile()
