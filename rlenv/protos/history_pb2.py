# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: history.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import pokemon_pb2 as pokemon__pb2
from . import enums_pb2 as enums__pb2
from . import messages_pb2 as messages__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rhistory.proto\x12\x07history\x1a\rpokemon.proto\x1a\x0b\x65nums.proto\x1a\x0emessages.proto\"8\n\x05\x42oost\x12 \n\x05index\x18\x01 \x01(\x0e\x32\x11.enums.BoostsEnum\x12\r\n\x05value\x18\x02 \x01(\x05\"H\n\rSidecondition\x12(\n\x05index\x18\x01 \x01(\x0e\x32\x19.enums.SideconditionsEnum\x12\r\n\x05value\x18\x02 \x01(\x05\"I\n\x0eVolatilestatus\x12(\n\x05index\x18\x01 \x01(\x0e\x32\x19.enums.VolatilestatusEnum\x12\r\n\x05value\x18\x02 \x01(\x05\"@\n\tHyphenArg\x12$\n\x05index\x18\x01 \x01(\x0e\x32\x15.enums.HyphenargsEnum\x12\r\n\x05value\x18\x02 \x01(\x08\"\xd8\x01\n\x0bHistorySide\x12 \n\x06\x61\x63tive\x18\x01 \x01(\x0b\x32\x10.pokemon.Pokemon\x12\x1e\n\x06\x62oosts\x18\x02 \x03(\x0b\x32\x0e.history.Boost\x12.\n\x0esideConditions\x18\x03 \x03(\x0b\x32\x16.history.Sidecondition\x12/\n\x0evolatileStatus\x18\x04 \x03(\x0b\x32\x17.history.Volatilestatus\x12&\n\nhyphenArgs\x18\x05 \x03(\x0b\x32\x12.history.HyphenArg\"\xf7\x01\n\x0bHistoryStep\x12 \n\x02p1\x18\x01 \x01(\x0b\x32\x14.history.HistorySide\x12 \n\x02p2\x18\x02 \x01(\x0b\x32\x14.history.HistorySide\x12$\n\x07weather\x18\x03 \x01(\x0e\x32\x13.enums.WeathersEnum\x12\x35\n\rpseudoweather\x18\x04 \x01(\x0b\x32\x1e.messages.PseudoweatherMessage\x12\'\n\x06\x61\x63tion\x18\x05 \x01(\x0e\x32\x17.history.ActionTypeEnum\x12\x1e\n\x04move\x18\x06 \x01(\x0e\x32\x10.enums.MovesEnum*&\n\x0e\x41\x63tionTypeEnum\x12\x08\n\x04move\x10\x00\x12\n\n\x06switch\x10\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'history_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ACTIONTYPEENUM']._serialized_start=812
  _globals['_ACTIONTYPEENUM']._serialized_end=850
  _globals['_BOOST']._serialized_start=70
  _globals['_BOOST']._serialized_end=126
  _globals['_SIDECONDITION']._serialized_start=128
  _globals['_SIDECONDITION']._serialized_end=200
  _globals['_VOLATILESTATUS']._serialized_start=202
  _globals['_VOLATILESTATUS']._serialized_end=275
  _globals['_HYPHENARG']._serialized_start=277
  _globals['_HYPHENARG']._serialized_end=341
  _globals['_HISTORYSIDE']._serialized_start=344
  _globals['_HISTORYSIDE']._serialized_end=560
  _globals['_HISTORYSTEP']._serialized_start=563
  _globals['_HISTORYSTEP']._serialized_end=810
# @@protoc_insertion_point(module_scope)