# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: servicev2.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "servicev2.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0fservicev2.proto\x12\tservicev2"\xc3\x01\n\rClientMessage\x12\x11\n\tplayer_id\x18\x01 \x01(\x05\x12\x0f\n\x07game_id\x18\x02 \x01(\x05\x12,\n\x07\x63onnect\x18\x03 \x01(\x0b\x32\x19.servicev2.ConnectMessageH\x00\x12&\n\x04step\x18\x04 \x01(\x0b\x32\x16.servicev2.StepMessageH\x00\x12(\n\x05reset\x18\x05 \x01(\x0b\x32\x17.servicev2.ResetMessageH\x00\x42\x0e\n\x0cmessage_type"3\n\x06\x41\x63tion\x12\x0c\n\x04rqid\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05\x12\x0c\n\x04text\x18\x03 \x01(\t"0\n\x0bStepMessage\x12!\n\x06\x61\x63tion\x18\x01 \x01(\x0b\x32\x11.servicev2.Action"\x0e\n\x0cResetMessage"\x10\n\x0e\x43onnectMessage"u\n\rServerMessage\x12*\n\ngame_state\x18\x01 \x01(\x0b\x32\x14.servicev2.GameStateH\x00\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x17.servicev2.ErrorMessageH\x00\x42\x0e\n\x0cmessage_type";\n\tGameState\x12\x11\n\tplayer_id\x18\x01 \x01(\x05\x12\x0c\n\x04rqid\x18\x02 \x01(\x05\x12\r\n\x05state\x18\x03 \x01(\x0c"%\n\x0c\x45rrorMessage\x12\x15\n\rerror_message\x18\x01 \x01(\tb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "servicev2_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_CLIENTMESSAGE"]._serialized_start = 31
    _globals["_CLIENTMESSAGE"]._serialized_end = 226
    _globals["_ACTION"]._serialized_start = 228
    _globals["_ACTION"]._serialized_end = 279
    _globals["_STEPMESSAGE"]._serialized_start = 281
    _globals["_STEPMESSAGE"]._serialized_end = 329
    _globals["_RESETMESSAGE"]._serialized_start = 331
    _globals["_RESETMESSAGE"]._serialized_end = 345
    _globals["_CONNECTMESSAGE"]._serialized_start = 347
    _globals["_CONNECTMESSAGE"]._serialized_end = 363
    _globals["_SERVERMESSAGE"]._serialized_start = 365
    _globals["_SERVERMESSAGE"]._serialized_end = 482
    _globals["_GAMESTATE"]._serialized_start = 484
    _globals["_GAMESTATE"]._serialized_end = 543
    _globals["_ERRORMESSAGE"]._serialized_start = 545
    _globals["_ERRORMESSAGE"]._serialized_end = 582
# @@protoc_insertion_point(module_scope)
