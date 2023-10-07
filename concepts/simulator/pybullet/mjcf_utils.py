#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mjcf_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle.io as io
import os.path as osp

__all__ = ['MJCFCanonizer', 'canonize_mjcf']


class MJCFCanonizer(object):
    def __init__(self, filename, xml_object=None):
        self.filename = filename
        self.dirname = osp.dirname(self.filename)
        self.xml_object = xml_object

        if self.xml_object is None:
            self.xml_object = io.load_xml(self.filename)

        self.directories = dict()

    def resolve_path(self, path, type=None):
        if osp.isfile(path):
            return path
        if type is not None:
            if type in self.directories:
                asset_path = osp.join(self.directories[type], path)
                if osp.isfile(asset_path):
                    return path
                else:
                    rel_path = osp.realpath(osp.join(self.dirname, path))
                    rel_path = osp.relpath(rel_path, start=self.directories[type])
                    return rel_path
        rel_path = osp.realpath(osp.join(self.dirname, path))
        return rel_path

    def _canonize_include_inner(self, object):
        found = False
        for k in list(object.keys()):
            if k == 'include':
                found = True
                v = object.pop(k)
                if isinstance(v, list):
                    for sub_v in v:
                        self._canonize_include_append(object, sub_v)
                else:
                    self._canonize_include_append(object, v)
            elif not k.startswith('__'):
                v = object[k]
                if isinstance(v, list):
                    for sub_v in v:
                        found = found or self._canonize_include_inner(sub_v)
                else:
                    found = found or self._canonize_include_inner(v)
        return found

    def _canonize_include_append(self, object, include_obj):
        sub_xml_object = io.load_xml(self.resolve_path(include_obj['__attribute__']['file']))
        for name, child in sub_xml_object.items():
            if not name.startswith('__'):
                _xml_add_to(object, name, child)

    def _canonize_include(self, object):
        for i in range(100):
            found = self._canonize_include_inner(object)
            if not found:
                break

    def _canonize_directories(self, object):
        if 'compiler' in object:
            attributes = object['compiler']['__attribute__']
            if 'meshdir' in attributes:
                self.directories['mesh'] = attributes['meshdir'] = self.resolve_path(attributes['meshdir']) + '/'
            if 'texturedir' in attributes:
                self.directories['texture'] = attributes['texturedir'] = (
                    self.resolve_path(attributes['texturedir']) + '/'
                )

    def _canonize_attributes(self, object):
        for k, v in object.items():
            if not k.startswith('__'):
                if isinstance(v, list):
                    for sub_v in v:
                        self._canonize_attributes(sub_v)
                else:
                    self._canonize_attributes(v)

        tag, attributes = object['__name__'], object['__attribute__']

        if tag == 'geom':
            if 'type' not in attributes:
                attributes['type'] = 'sphere'
        elif tag == 'joint':
            if 'type' not in attributes:
                attributes['type'] = 'hinge'
        elif tag == 'mesh':
            if 'file' in attributes:
                attributes['file'] = self.resolve_path(attributes['file'], type='mesh')
        elif tag == 'texture':
            if 'file' in attributes:
                attributes['file'] = self.resolve_path(attributes['file'], type='texture')

    def canonize(self):
        self._canonize_include(self.xml_object)
        self._canonize_directories(self.xml_object)
        self._canonize_attributes(self.xml_object)
        return self.xml_object


def canonize_mjcf(filename, xml_object=None):
    return MJCFCanonizer(filename, xml_object=xml_object).canonize()


def _xml_add_to(object, name, child):
    if name not in object:
        object[name] = child
    else:
        if isinstance(object[name], list):
            if isinstance(child, list):
                object[name].extend(child)
            else:
                object[name].append(child)
        else:
            if isinstance(child, list):
                object[name] = [object[name]] + child
            else:
                object[name] = [object[name], child]
    return object
