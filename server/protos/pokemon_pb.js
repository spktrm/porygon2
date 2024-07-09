// source: pokemon.proto
/**
 * @fileoverview
 * @enhanceable
 * @suppress {missingRequire} reports error on implicit type usages.
 * @suppress {messageConventions} JS Compiler reports an error if a variable or
 *     field starts with 'MSG_' and isn't a translatable message.
 * @public
 */
// GENERATED CODE -- DO NOT EDIT!
/* eslint-disable */
// @ts-nocheck

var jspb = require('google-protobuf');
var goog = jspb;
var global = (function() { return this || window || global || self || Function('return this')(); }).call(null);

var enums_pb = require('./enums_pb.js');
goog.object.extend(proto, enums_pb);
goog.exportSymbol('proto.pokemon.Pokemon', null, global);
/**
 * Generated by JsPbCodeGenerator.
 * @param {Array=} opt_data Optional initial data array, typically from a
 * server response, or constructed directly in Javascript. The array is used
 * in place and becomes part of the constructed object. It is not cloned.
 * If no data is provided, the constructed object will be empty, but still
 * valid.
 * @extends {jspb.Message}
 * @constructor
 */
proto.pokemon.Pokemon = function(opt_data) {
  jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.pokemon.Pokemon, jspb.Message);
if (goog.DEBUG && !COMPILED) {
  /**
   * @public
   * @override
   */
  proto.pokemon.Pokemon.displayName = 'proto.pokemon.Pokemon';
}



if (jspb.Message.GENERATE_TO_OBJECT) {
/**
 * Creates an object representation of this proto.
 * Field names that are reserved in JavaScript and will be renamed to pb_name.
 * Optional fields that are not set will be set to undefined.
 * To access a reserved field use, foo.pb_<name>, eg, foo.pb_default.
 * For the list of reserved names please see:
 *     net/proto2/compiler/js/internal/generator.cc#kKeyword.
 * @param {boolean=} opt_includeInstance Deprecated. whether to include the
 *     JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @return {!Object}
 */
proto.pokemon.Pokemon.prototype.toObject = function(opt_includeInstance) {
  return proto.pokemon.Pokemon.toObject(opt_includeInstance, this);
};


/**
 * Static version of the {@see toObject} method.
 * @param {boolean|undefined} includeInstance Deprecated. Whether to include
 *     the JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @param {!proto.pokemon.Pokemon} msg The msg instance to transform.
 * @return {!Object}
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.pokemon.Pokemon.toObject = function(includeInstance, msg) {
  var f, obj = {
    species: jspb.Message.getFieldWithDefault(msg, 1, 0),
    item: jspb.Message.getFieldWithDefault(msg, 2, 0),
    ability: jspb.Message.getFieldWithDefault(msg, 3, 0),
    move1id: jspb.Message.getFieldWithDefault(msg, 4, 0),
    move2id: jspb.Message.getFieldWithDefault(msg, 5, 0),
    move3id: jspb.Message.getFieldWithDefault(msg, 6, 0),
    move4id: jspb.Message.getFieldWithDefault(msg, 7, 0),
    pp1used: jspb.Message.getFieldWithDefault(msg, 8, 0),
    pp2used: jspb.Message.getFieldWithDefault(msg, 9, 0),
    pp3used: jspb.Message.getFieldWithDefault(msg, 10, 0),
    pp4used: jspb.Message.getFieldWithDefault(msg, 11, 0),
    hpratio: jspb.Message.getFloatingPointFieldWithDefault(msg, 12, 0.0),
    active: jspb.Message.getBooleanFieldWithDefault(msg, 13, false),
    fainted: jspb.Message.getBooleanFieldWithDefault(msg, 14, false),
    level: jspb.Message.getFieldWithDefault(msg, 15, 0),
    gender: jspb.Message.getFieldWithDefault(msg, 16, 0),
    itemeffect: jspb.Message.getFieldWithDefault(msg, 17, 0),
    hp: jspb.Message.getFieldWithDefault(msg, 18, 0),
    maxhp: jspb.Message.getFieldWithDefault(msg, 19, 0),
    status: jspb.Message.getFieldWithDefault(msg, 20, 0)
  };

  if (includeInstance) {
    obj.$jspbMessageInstance = msg;
  }
  return obj;
};
}


/**
 * Deserializes binary data (in protobuf wire format).
 * @param {jspb.ByteSource} bytes The bytes to deserialize.
 * @return {!proto.pokemon.Pokemon}
 */
proto.pokemon.Pokemon.deserializeBinary = function(bytes) {
  var reader = new jspb.BinaryReader(bytes);
  var msg = new proto.pokemon.Pokemon;
  return proto.pokemon.Pokemon.deserializeBinaryFromReader(msg, reader);
};


/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.pokemon.Pokemon} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.pokemon.Pokemon}
 */
proto.pokemon.Pokemon.deserializeBinaryFromReader = function(msg, reader) {
  while (reader.nextField()) {
    if (reader.isEndGroup()) {
      break;
    }
    var field = reader.getFieldNumber();
    switch (field) {
    case 1:
      var value = /** @type {!proto.enums.SpeciesEnum} */ (reader.readEnum());
      msg.setSpecies(value);
      break;
    case 2:
      var value = /** @type {!proto.enums.ItemsEnum} */ (reader.readEnum());
      msg.setItem(value);
      break;
    case 3:
      var value = /** @type {!proto.enums.AbilitiesEnum} */ (reader.readEnum());
      msg.setAbility(value);
      break;
    case 4:
      var value = /** @type {!proto.enums.MovesEnum} */ (reader.readEnum());
      msg.setMove1id(value);
      break;
    case 5:
      var value = /** @type {!proto.enums.MovesEnum} */ (reader.readEnum());
      msg.setMove2id(value);
      break;
    case 6:
      var value = /** @type {!proto.enums.MovesEnum} */ (reader.readEnum());
      msg.setMove3id(value);
      break;
    case 7:
      var value = /** @type {!proto.enums.MovesEnum} */ (reader.readEnum());
      msg.setMove4id(value);
      break;
    case 8:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setPp1used(value);
      break;
    case 9:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setPp2used(value);
      break;
    case 10:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setPp3used(value);
      break;
    case 11:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setPp4used(value);
      break;
    case 12:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setHpratio(value);
      break;
    case 13:
      var value = /** @type {boolean} */ (reader.readBool());
      msg.setActive(value);
      break;
    case 14:
      var value = /** @type {boolean} */ (reader.readBool());
      msg.setFainted(value);
      break;
    case 15:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setLevel(value);
      break;
    case 16:
      var value = /** @type {!proto.enums.GendersEnum} */ (reader.readEnum());
      msg.setGender(value);
      break;
    case 17:
      var value = /** @type {!proto.enums.ItemeffectEnum} */ (reader.readEnum());
      msg.setItemeffect(value);
      break;
    case 18:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setHp(value);
      break;
    case 19:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setMaxhp(value);
      break;
    case 20:
      var value = /** @type {!proto.enums.StatusesEnum} */ (reader.readEnum());
      msg.setStatus(value);
      break;
    default:
      reader.skipField();
      break;
    }
  }
  return msg;
};


/**
 * Serializes the message to binary data (in protobuf wire format).
 * @return {!Uint8Array}
 */
proto.pokemon.Pokemon.prototype.serializeBinary = function() {
  var writer = new jspb.BinaryWriter();
  proto.pokemon.Pokemon.serializeBinaryToWriter(this, writer);
  return writer.getResultBuffer();
};


/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.pokemon.Pokemon} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.pokemon.Pokemon.serializeBinaryToWriter = function(message, writer) {
  var f = undefined;
  f = message.getSpecies();
  if (f !== 0.0) {
    writer.writeEnum(
      1,
      f
    );
  }
  f = message.getItem();
  if (f !== 0.0) {
    writer.writeEnum(
      2,
      f
    );
  }
  f = message.getAbility();
  if (f !== 0.0) {
    writer.writeEnum(
      3,
      f
    );
  }
  f = message.getMove1id();
  if (f !== 0.0) {
    writer.writeEnum(
      4,
      f
    );
  }
  f = message.getMove2id();
  if (f !== 0.0) {
    writer.writeEnum(
      5,
      f
    );
  }
  f = message.getMove3id();
  if (f !== 0.0) {
    writer.writeEnum(
      6,
      f
    );
  }
  f = message.getMove4id();
  if (f !== 0.0) {
    writer.writeEnum(
      7,
      f
    );
  }
  f = message.getPp1used();
  if (f !== 0) {
    writer.writeInt32(
      8,
      f
    );
  }
  f = message.getPp2used();
  if (f !== 0) {
    writer.writeInt32(
      9,
      f
    );
  }
  f = message.getPp3used();
  if (f !== 0) {
    writer.writeInt32(
      10,
      f
    );
  }
  f = message.getPp4used();
  if (f !== 0) {
    writer.writeInt32(
      11,
      f
    );
  }
  f = message.getHpratio();
  if (f !== 0.0) {
    writer.writeFloat(
      12,
      f
    );
  }
  f = message.getActive();
  if (f) {
    writer.writeBool(
      13,
      f
    );
  }
  f = message.getFainted();
  if (f) {
    writer.writeBool(
      14,
      f
    );
  }
  f = message.getLevel();
  if (f !== 0) {
    writer.writeInt32(
      15,
      f
    );
  }
  f = message.getGender();
  if (f !== 0.0) {
    writer.writeEnum(
      16,
      f
    );
  }
  f = message.getItemeffect();
  if (f !== 0.0) {
    writer.writeEnum(
      17,
      f
    );
  }
  f = message.getHp();
  if (f !== 0) {
    writer.writeInt32(
      18,
      f
    );
  }
  f = message.getMaxhp();
  if (f !== 0) {
    writer.writeInt32(
      19,
      f
    );
  }
  f = message.getStatus();
  if (f !== 0.0) {
    writer.writeEnum(
      20,
      f
    );
  }
};


/**
 * optional enums.SpeciesEnum species = 1;
 * @return {!proto.enums.SpeciesEnum}
 */
proto.pokemon.Pokemon.prototype.getSpecies = function() {
  return /** @type {!proto.enums.SpeciesEnum} */ (jspb.Message.getFieldWithDefault(this, 1, 0));
};


/**
 * @param {!proto.enums.SpeciesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setSpecies = function(value) {
  return jspb.Message.setProto3EnumField(this, 1, value);
};


/**
 * optional enums.ItemsEnum item = 2;
 * @return {!proto.enums.ItemsEnum}
 */
proto.pokemon.Pokemon.prototype.getItem = function() {
  return /** @type {!proto.enums.ItemsEnum} */ (jspb.Message.getFieldWithDefault(this, 2, 0));
};


/**
 * @param {!proto.enums.ItemsEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setItem = function(value) {
  return jspb.Message.setProto3EnumField(this, 2, value);
};


/**
 * optional enums.AbilitiesEnum ability = 3;
 * @return {!proto.enums.AbilitiesEnum}
 */
proto.pokemon.Pokemon.prototype.getAbility = function() {
  return /** @type {!proto.enums.AbilitiesEnum} */ (jspb.Message.getFieldWithDefault(this, 3, 0));
};


/**
 * @param {!proto.enums.AbilitiesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setAbility = function(value) {
  return jspb.Message.setProto3EnumField(this, 3, value);
};


/**
 * optional enums.MovesEnum move1Id = 4;
 * @return {!proto.enums.MovesEnum}
 */
proto.pokemon.Pokemon.prototype.getMove1id = function() {
  return /** @type {!proto.enums.MovesEnum} */ (jspb.Message.getFieldWithDefault(this, 4, 0));
};


/**
 * @param {!proto.enums.MovesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setMove1id = function(value) {
  return jspb.Message.setProto3EnumField(this, 4, value);
};


/**
 * optional enums.MovesEnum move2Id = 5;
 * @return {!proto.enums.MovesEnum}
 */
proto.pokemon.Pokemon.prototype.getMove2id = function() {
  return /** @type {!proto.enums.MovesEnum} */ (jspb.Message.getFieldWithDefault(this, 5, 0));
};


/**
 * @param {!proto.enums.MovesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setMove2id = function(value) {
  return jspb.Message.setProto3EnumField(this, 5, value);
};


/**
 * optional enums.MovesEnum move3Id = 6;
 * @return {!proto.enums.MovesEnum}
 */
proto.pokemon.Pokemon.prototype.getMove3id = function() {
  return /** @type {!proto.enums.MovesEnum} */ (jspb.Message.getFieldWithDefault(this, 6, 0));
};


/**
 * @param {!proto.enums.MovesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setMove3id = function(value) {
  return jspb.Message.setProto3EnumField(this, 6, value);
};


/**
 * optional enums.MovesEnum move4Id = 7;
 * @return {!proto.enums.MovesEnum}
 */
proto.pokemon.Pokemon.prototype.getMove4id = function() {
  return /** @type {!proto.enums.MovesEnum} */ (jspb.Message.getFieldWithDefault(this, 7, 0));
};


/**
 * @param {!proto.enums.MovesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setMove4id = function(value) {
  return jspb.Message.setProto3EnumField(this, 7, value);
};


/**
 * optional int32 pp1Used = 8;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getPp1used = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 8, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setPp1used = function(value) {
  return jspb.Message.setProto3IntField(this, 8, value);
};


/**
 * optional int32 pp2Used = 9;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getPp2used = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 9, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setPp2used = function(value) {
  return jspb.Message.setProto3IntField(this, 9, value);
};


/**
 * optional int32 pp3Used = 10;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getPp3used = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 10, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setPp3used = function(value) {
  return jspb.Message.setProto3IntField(this, 10, value);
};


/**
 * optional int32 pp4Used = 11;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getPp4used = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 11, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setPp4used = function(value) {
  return jspb.Message.setProto3IntField(this, 11, value);
};


/**
 * optional float hpRatio = 12;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getHpratio = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 12, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setHpratio = function(value) {
  return jspb.Message.setProto3FloatField(this, 12, value);
};


/**
 * optional bool active = 13;
 * @return {boolean}
 */
proto.pokemon.Pokemon.prototype.getActive = function() {
  return /** @type {boolean} */ (jspb.Message.getBooleanFieldWithDefault(this, 13, false));
};


/**
 * @param {boolean} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setActive = function(value) {
  return jspb.Message.setProto3BooleanField(this, 13, value);
};


/**
 * optional bool fainted = 14;
 * @return {boolean}
 */
proto.pokemon.Pokemon.prototype.getFainted = function() {
  return /** @type {boolean} */ (jspb.Message.getBooleanFieldWithDefault(this, 14, false));
};


/**
 * @param {boolean} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setFainted = function(value) {
  return jspb.Message.setProto3BooleanField(this, 14, value);
};


/**
 * optional int32 level = 15;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getLevel = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 15, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setLevel = function(value) {
  return jspb.Message.setProto3IntField(this, 15, value);
};


/**
 * optional enums.GendersEnum gender = 16;
 * @return {!proto.enums.GendersEnum}
 */
proto.pokemon.Pokemon.prototype.getGender = function() {
  return /** @type {!proto.enums.GendersEnum} */ (jspb.Message.getFieldWithDefault(this, 16, 0));
};


/**
 * @param {!proto.enums.GendersEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setGender = function(value) {
  return jspb.Message.setProto3EnumField(this, 16, value);
};


/**
 * optional enums.ItemeffectEnum itemEffect = 17;
 * @return {!proto.enums.ItemeffectEnum}
 */
proto.pokemon.Pokemon.prototype.getItemeffect = function() {
  return /** @type {!proto.enums.ItemeffectEnum} */ (jspb.Message.getFieldWithDefault(this, 17, 0));
};


/**
 * @param {!proto.enums.ItemeffectEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setItemeffect = function(value) {
  return jspb.Message.setProto3EnumField(this, 17, value);
};


/**
 * optional int32 hp = 18;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getHp = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 18, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setHp = function(value) {
  return jspb.Message.setProto3IntField(this, 18, value);
};


/**
 * optional int32 maxHp = 19;
 * @return {number}
 */
proto.pokemon.Pokemon.prototype.getMaxhp = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 19, 0));
};


/**
 * @param {number} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setMaxhp = function(value) {
  return jspb.Message.setProto3IntField(this, 19, value);
};


/**
 * optional enums.StatusesEnum status = 20;
 * @return {!proto.enums.StatusesEnum}
 */
proto.pokemon.Pokemon.prototype.getStatus = function() {
  return /** @type {!proto.enums.StatusesEnum} */ (jspb.Message.getFieldWithDefault(this, 20, 0));
};


/**
 * @param {!proto.enums.StatusesEnum} value
 * @return {!proto.pokemon.Pokemon} returns this
 */
proto.pokemon.Pokemon.prototype.setStatus = function(value) {
  return jspb.Message.setProto3EnumField(this, 20, value);
};


goog.object.extend(exports, proto.pokemon);
