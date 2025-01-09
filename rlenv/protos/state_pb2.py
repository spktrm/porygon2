# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: state.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'state.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import history_pb2 as history__pb2
from . import enums_pb2 as enums__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bstate.proto\x12\x05rlenv\x1a\rhistory.proto\x1a\x0b\x65nums.proto\"\x8e\x01\n\x07Rewards\x12\x11\n\twinReward\x18\x01 \x01(\x02\x12\x10\n\x08hpReward\x18\x02 \x01(\x02\x12\x15\n\rfaintedReward\x18\x03 \x01(\x02\x12\x14\n\x0cswitchReward\x18\x04 \x01(\x02\x12\x17\n\x0flongevityReward\x18\x05 \x01(\x02\x12\x18\n\x10\x66\x61intedFibReward\x18\x06 \x01(\x02\"%\n\nHeuristics\x12\x17\n\x0fheuristicAction\x18\x01 \x01(\x05\"\xf5\x01\n\x04Info\x12\x0e\n\x06gameId\x18\x01 \x01(\x05\x12\x0c\n\x04\x64one\x18\x02 \x01(\x08\x12\x13\n\x0bplayerIndex\x18\x03 \x01(\x08\x12\x0c\n\x04turn\x18\x04 \x01(\x05\x12\n\n\x02ts\x18\x05 \x01(\x02\x12\x11\n\tdrawRatio\x18\x06 \x01(\x02\x12\x13\n\x0bworkerIndex\x18\x07 \x01(\x05\x12\x1f\n\x07rewards\x18\x08 \x01(\x0b\x32\x0e.rlenv.Rewards\x12\x0c\n\x04seed\x18\t \x01(\x05\x12\x0c\n\x04\x64raw\x18\n \x01(\x08\x12%\n\nheuristics\x18\x0b \x01(\x0b\x32\x11.rlenv.Heuristics\x12\x14\n\x0crequestCount\x18\x0c \x01(\x05\"\x87\x01\n\x05State\x12\x19\n\x04info\x18\x01 \x01(\x0b\x32\x0b.rlenv.Info\x12\x14\n\x0clegalActions\x18\x02 \x01(\x0c\x12!\n\x07history\x18\x03 \x01(\x0b\x32\x10.history.History\x12\x0f\n\x07moveset\x18\x04 \x01(\x0c\x12\x0c\n\x04team\x18\x05 \x01(\x0c\x12\x0b\n\x03key\x18\x06 \x01(\t\"L\n\nTrajectory\x12\x1c\n\x06states\x18\x01 \x03(\x0b\x32\x0c.rlenv.State\x12\x0f\n\x07\x61\x63tions\x18\x02 \x03(\x05\x12\x0f\n\x07rewards\x18\x03 \x03(\x05\"2\n\x07\x44\x61taset\x12\'\n\x0ctrajectories\x18\x01 \x03(\x0b\x32\x11.rlenv.Trajectoryb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'state_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_REWARDS']._serialized_start=51
  _globals['_REWARDS']._serialized_end=193
  _globals['_HEURISTICS']._serialized_start=195
  _globals['_HEURISTICS']._serialized_end=232
  _globals['_INFO']._serialized_start=235
  _globals['_INFO']._serialized_end=480
  _globals['_STATE']._serialized_start=483
  _globals['_STATE']._serialized_end=618
  _globals['_TRAJECTORY']._serialized_start=620
  _globals['_TRAJECTORY']._serialized_end=696
  _globals['_DATASET']._serialized_start=698
  _globals['_DATASET']._serialized_end=748
# @@protoc_insertion_point(module_scope)
