from PIL import Image, ImageFont, ImageDraw
import pandas as pd

def print_tree(tree, 
               name='tree', 
               node_size=50, 
               node_distance=80, 
               line_step=200):
    
    def print_tree_values(node, 
                          linear_tree=None, 
                          depth=None):
        if depth is None:
            depth = 0
        else:
            depth += 1

        if linear_tree is None:
            linear_tree = []

        linear_tree.append([node, depth])

        for child in node.childs:
            print_tree_values(child, linear_tree, depth)

        return linear_tree
    
    linear_tree = print_tree_values(tree)
    
    linear_nodes_df = pd.DataFrame(linear_tree, columns=['node','depth'])
    
    max_depth = linear_nodes_df['depth'].max() + 1
    max_width = linear_nodes_df.groupby('depth').count()['node'].max()
    number_of_nodes_by_depth = linear_nodes_df.groupby('depth').count()
    
    margin = 100

    stage_width = ((node_distance + node_size) * max_width) + margin
    stage_height = (line_step * max_depth)

    image = Image.new('RGB', (stage_width+margin, stage_height), color = 'white')
    draw = ImageDraw.Draw(image)
    
    # Print the node, the conexion lines and the text
    def print_node(draw, x, y, l, w, cl, father_x=0, father_y=0, text=None):
        draw.ellipse((x, y, l+cl, w+cl), 'white', 'black')
        
        if text is not None:
            draw.text((x+10, y+20), text, fill=(0,0,0))
            
        if father_x > 0:
            draw.line((x+cl, y, father_x+cl, father_y+cl*2), fill=128)
            
    print_depth = {}
    depth_father = {}
    
    # Loop to print the data
    for index, row in linear_nodes_df.iterrows():
        if print_depth.get(row.depth) is None:
            print_depth[row.depth] = 1
        else:
            print_depth[row.depth] += 1
        
        num_nodes_depth = print_depth.get(row.depth)
        num_max_on_depth = number_of_nodes_by_depth.iloc[row.depth].node

        width_correction = (stage_width/num_max_on_depth)/2
        node_pos_x = ((stage_width/num_max_on_depth) * num_nodes_depth) -width_correction
        
        node_pos_y = line_step * row.depth

        node_pos_x_end = node_pos_x + node_size
        node_pos_y_end = node_pos_y + node_size

        if row.depth > 0:
            father = depth_father.get(row.depth-1)
            father_x = father[0]
            father_y = father[1]
        else:
            father_x = 0
            father_y = 0

        print_node(draw,
                   node_pos_x,
                   node_pos_y,
                   node_pos_x_end,
                   node_pos_y_end,
                   node_size,
                   father_x, 
                   father_y,
                   f"{row.node.attribute} \n{row.node.attribute_value} \n{row.node.probs}")

        depth_father[row.depth] = [node_pos_x, node_pos_y]

    image.save(f'{name}.png')