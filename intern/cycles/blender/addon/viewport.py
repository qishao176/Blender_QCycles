# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Native viewport passes, resident LPE caches, and on-demand LPE sampling."""

import bpy
from bpy.app.translations import pgettext_iface as tr

# RNA dynamic enum strings and numeric IDs must outlive every menu invocation.
_names = {}
_values = {}
_items = {}


def _id(name):
    if name not in _values:
        value = len(_values) + 1
        _values[name] = value
        _names[value] = name
    return _values[name]


def catalog(context):
    import _cycles
    layer = getattr(context, 'view_layer', None)
    tree = layer.lpe_node_tree if layer else None
    return dict(_cycles.lpe_viewport_catalog(tree.as_pointer())) if tree else {}


def display_items(self, context):
    from .properties import enum_view3d_shading_render_pass, enum_view3d_debug_render_pass
    result = []
    native = enum_view3d_shading_render_pass
    if context and context.preferences.experimental.use_cycles_debug and context.preferences.view.show_developer_ui:
        native += enum_view3d_debug_render_pass
    for identifier, name, description in native:
        if not identifier:
            result.append(('', name, ''))
            continue
        key = 'N:' + identifier
        result.append(_items.setdefault(key, (key, name, description, 0, _id(key))))
    outputs = catalog(context) if context else {}
    names = [name for name, resident in outputs.items() if not resident]
    if names:
        result.append(('', "LPE Channels — Uncached", ''))
        description = "Render this Render Only output on demand; switching to it restarts sampling"
        for name in names:
            result.append(_items.setdefault((name, False), (
                name, name, description, 0, _id(name))))
    return result


def link_label(link):
    source = link.from_node
    if source.bl_idname == 'LPENodeSceneEdit':
        # Each output mirrors its input item, including its editable name.
        return link.from_socket.name
    return f"{source.label or source.name} · {tr(link.from_socket.name)}"


def overlay_items(self, context):
    result = [_items.setdefault('OFF', ('OFF', "None", "", 0, _id('OFF')))]
    if not context:
        return result
    outputs = catalog(context)
    tree = context.view_layer.lpe_node_tree
    for name, cached in outputs.items():
        if cached and not name.startswith('E:'):
            result.append(_items.setdefault((name, True), (name, name, "", 0, _id(name))))
    if any(name.startswith('E:') for name in outputs):
        result.append(('', "Light Edit", ''))
        for node in tree.nodes:
            if node.bl_idname != 'LPENodeSceneEdit':
                continue
            for socket in node.inputs:
                key = f"E:{node.node_identifier}:{socket.identifier}"
                if key not in outputs:
                    continue
                label = socket.name
                result.append(_items.setdefault((key, label), (key, label, "", 0, _id(key))))
    return result


def overlay_get(self):
    return _id(self.lpe_overlay_output or 'OFF')


def overlay_set(self, value):
    name = _names.get(value)
    if name:
        self.lpe_overlay_output = '' if name == 'OFF' else name


def display_get(self):
    return _id(self.lpe_display_output or 'N:' + self.render_pass)


def _choose(self, name):
    if name.startswith('N:'):
        self.lpe_display_output = ''
        if self.render_pass != name[2:]:
            self.render_pass = name[2:]
    else:
        self.lpe_display_output = name


def display_set(self, value):
    name = _names.get(value)
    if name:
        self.lpe_overlay_output = ''
        _choose(self, name)


def publish_display(settings, context, viewer_link=None):
    import _cycles
    base = settings.lpe_display_output
    native = settings.render_pass
    if base:
        label = base
    else:
        item = settings.bl_rna.properties['render_pass'].enum_items.get(native)
        label = tr(item.name) if item else native
    status = f"{tr('Render Pass')} · {label}"
    overlay = settings.lpe_overlay_output
    if overlay:
        label = overlay
        tree = context.view_layer.lpe_node_tree
        if overlay == 'LPE Viewer':
            label = tr('LPE Viewer')
            if viewer_link:
                label += f" ({link_label(viewer_link)})"
        elif overlay.startswith('E:') and tree:
            _, identifier, socket_id = overlay.split(':', 2)
            node = next((n for n in tree.nodes
                         if n.bl_idname == 'LPENodeSceneEdit' and n.node_identifier == int(identifier)), None)
            socket = next((s for s in node.inputs if s.identifier == socket_id), None) if node else None
            label = socket.name if socket else tr('Light Edit')
        status += f" / {tr('LPE Preview Overlay')} · {label}"
    _cycles.lpe_viewport_select(context.space_data.as_pointer(), overlay or base or 'N:' + native,
                                bool(overlay), status)


def display_update(self, context):
    space = getattr(context, 'space_data', None)
    if space and space.type == 'VIEW_3D':
        publish_display(self, context)
        context.area.tag_redraw()


def sync_viewer(context):
    """Auto-select on Viewer activation/reconnection; allow later manual choices."""
    from .engine import use_vcm
    space = context.space_data
    if not space or space.type != 'VIEW_3D' or use_vcm(context.scene, context.view_layer):
        return
    settings = space.shading.cycles
    tree = context.view_layer.lpe_node_tree
    signature = ''
    viewer_link = None
    if tree:
        for node in tree.nodes:
            if node.bl_idname != 'LPENodeViewer' or node.mute or not node.active:
                continue
            links = [link for link in node.inputs[0].links if not link.is_muted]
            if links:
                link = links[0]
                viewer_link = link
                signature = f"{tree.session_uid}:{node.as_pointer()}:{link.from_node.as_pointer()}:{link.from_socket.identifier}"
                break
    if signature != settings.lpe_viewer_signature:
        if signature:
            if settings.lpe_overlay_output != 'LPE Viewer':
                settings.lpe_previous_display = settings.lpe_overlay_output
            settings.lpe_overlay_output = 'LPE Viewer'
        elif settings.lpe_overlay_output == 'LPE Viewer':
            settings.lpe_overlay_output = settings.lpe_previous_display
        settings.lpe_viewer_signature = signature
    if settings.lpe_overlay_output or settings.lpe_display_output:
        outputs = catalog(context)
        if settings.lpe_overlay_output and not outputs.get(settings.lpe_overlay_output):
            settings.lpe_overlay_output = ''
        if settings.lpe_display_output and outputs.get(settings.lpe_display_output) is not False:
            settings.lpe_display_output = ''
    publish_display(settings, context, viewer_link)
