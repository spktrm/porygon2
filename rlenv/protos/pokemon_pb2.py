# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pokemon.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import enums_pb2 as enums__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rpokemon.proto\x12\x07pokemon\x1a\x0b\x65nums.proto\"8\n\x04Move\x12 \n\x06moveId\x18\x01 \x01(\x0e\x32\x10.enums.MovesEnum\x12\x0e\n\x06ppUsed\x18\x02 \x01(\x05\"\xfa\x01\n\x07Pokemon\x12#\n\x07species\x18\x01 \x01(\x0e\x32\x12.enums.SpeciesEnum\x12\x1e\n\x04item\x18\x02 \x01(\x0e\x32\x10.enums.ItemsEnum\x12%\n\x07\x61\x62ility\x18\x03 \x01(\x0e\x32\x14.enums.AbilitiesEnum\x12\x1e\n\x07moveset\x18\x04 \x03(\x0b\x32\r.pokemon.Move\x12\x0f\n\x07hpRatio\x18\x05 \x01(\x02\x12\x0e\n\x06\x61\x63tive\x18\x06 \x01(\x08\x12\x0f\n\x07\x66\x61inted\x18\x07 \x01(\x08\x12\r\n\x05level\x18\x08 \x01(\x05\x12\"\n\x06gender\x18\t \x01(\x0e\x32\x12.enums.GendersEnumb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'pokemon_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_MOVE']._serialized_start=39
  _globals['_MOVE']._serialized_end=95
  _globals['_POKEMON']._serialized_start=98
  _globals['_POKEMON']._serialized_end=348
# @@protoc_insertion_point(module_scope)
